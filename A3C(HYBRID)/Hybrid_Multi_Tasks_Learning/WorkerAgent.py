import numpy as np
import tensorflow as tf
from Neural_Network import Create_Network
#Initializing stack with 4 images of environment reset
def Stack_Frames(Frame):
    return np.stack([Frame]*4, axis=2)
#throw the oldest image and stack the newest image
def Shift_Frames(state, next_frame):
    return np.append(state[:, :, 1:], np.expand_dims(next_frame, 2), axis=2)
#Copy Parameters one scope to another scope
def Copy_Params_Scopes(sour_vars,dist_vars):

    src_vars = list(sorted(sour_vars, key=lambda v: v.name))
    dst_vars = list(sorted(dist_vars, key=lambda v: v.name))
    parmeters = []
    for source, dist in zip(src_vars, dst_vars):
        op = dist.assign(source)
        parmeters.append(op)
    return parmeters
#Use Gradients from local network to update the global networks
def Make_Training_Operation(Local_net, Global_net):

    Local_grads, _ = zip(*Local_net.grads_and_vars)
    Local_grads, _ = tf.clip_by_global_norm(Local_grads, 5.0)
    _, Global_vars = zip(*Global_net.grads_and_vars)
    l_grads_g_vars = list(zip(Local_grads, Global_vars))

    return Global_net.optimizer.apply_gradients(l_grads_g_vars, global_step=tf.compat.v1.train.get_global_step()),

#A tuple of one step
class Batch:
    def __init__(self, state, action, reward, next_state, done,value):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.value=value

#Transformation high resolution images for input into neural network
class Preprocesing_Input:
    def __init__(self):
        with tf.compat.v1.variable_scope("Image_Transformer"):
            self.input_image = tf.compat.v1.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output_image = tf.image.rgb_to_grayscale(self.input_image)
            self.output_image = tf.image.crop_to_bounding_box(self.output_image, 34, 0, 160, 160)
            self.output_image = tf.image.resize(self.output_image, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output_image = tf.squeeze(self.output_image)
    def transfrom(self, state, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        return sess.run(self.output_image, {self.input_image: state})

class Worker:
    def __init__(self, worker_name,
                 env,
                 global_policy,
                 global_value,
                 returns_list,
                 steps_list,
                 global_counter,
                 discount_factor=0.99,
                 max_global_steps=None):
        self.worker_name = worker_name
        self.env = env
        self.global_policy = global_policy
        self.global_value=global_value
        self.global_counter = global_counter
        self.max_global_steps = max_global_steps
        self.discount_factor = discount_factor
        self.Preprocessing = Preprocesing_Input()
        self.steps_list = steps_list
        self.a = 0

        with tf.compat.v1.variable_scope(self.worker_name):
            self.local_policy,self.local_value = Create_Network(global_policy.num_of_output)

        self.copy_params = Copy_Params_Scopes(
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="Global"),
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.worker_name+"/"))
        self.Value_Train = Make_Training_Operation(self.local_value, self.global_value)
        self.Policy_Train = Make_Training_Operation(self.local_policy,self.global_policy)
        self.hct = [np.zeros((1, 256)), np.zeros((1, 256))]
        self.vhct = [np.zeros((1, 256)), np.zeros((1, 256))]
        self.state = None
        self.total_reward = 0.
        self.step=0
        self.returns_list = returns_list



    def Run(self, sess, coord, t_max):
        with sess.as_default(), sess.graph.as_default():
            self.state = Stack_Frames(self.Preprocessing.transfrom(self.env.reset()))

            try:
                while not coord.should_stop():
                    self.hct = [np.zeros((1, 256)), np.zeros((1, 256))]
                    self.vhct = [np.zeros((1, 256)), np.zeros((1, 256))]
                    sess.run(self.copy_params)
                    steps, global_step= self.Run_N_Steps(t_max, sess)
                    if self.max_global_steps is not None and global_step >= self.max_global_steps:
                        coord.request_stop()
                        return
                    self.Update_Network(steps, sess)

                    #if(steps[-1].done):

            except tf.errors.CancelledError:
                return
    def Run_N_Steps(self, num, sess):
        steps = []

        for _ in range(num):
            action, self.hct = self.Select_Action(self.state, sess,self.hct)
            value, self.vhct = self.Select_value(self.state, sess, self.vhct)
            next_frame, reward, done, _ = self.env.step(action)
            next_state = Shift_Frames(self.state, self.Preprocessing.transfrom(next_frame))
            if done:
                print("Total reward:", self.total_reward, "Worker:", self.worker_name,"step:",self.step)
                self.step = 0
                self.returns_list.append(self.total_reward)
                if len(self.returns_list) > 0 and len(self.returns_list) % 100 == 0:
                    print("**** Total average reward (last 100):", np.mean(self.returns_list[-100:]), "Total episode:", len(self.returns_list), "Count:", self.global_counter)
                self.total_reward = 0.
                self.steps_list.append(self.a)
            else:
                self.total_reward += reward
            self.step+=1
            step = Batch(self.state, action, reward, next_state, done,value)
            steps.append(step)
            global_step = next(self.global_counter)
            self.a=global_step
            if done:
                self.state = Stack_Frames(self.Preprocessing.transfrom(self.env.reset()))
                break
            else:
                self.state = next_state
        return steps, global_step
    def Select_Action(self, state, sess,phct):
        feed_dict = {self.local_policy.states: [state],self.local_policy.ht:phct[0],self.local_policy.ct:phct[1]}
        actions,hct= sess.run([self.local_policy.sample_action,self.local_policy.state_out], feed_dict)
        return actions[0],hct
    def Select_value(self, state, sess,vhct):
        feed_dict = {self.local_value.vstates: [state],self.local_value.vht:vhct[0],self.local_value.vct:vhct[1]}
        vhat,vhct= sess.run([ self.local_value.vhat,self.local_value.vstate_out], feed_dict)
        return vhat[0],vhct

    def Update_Network(self, steps, sess):
        reward = 0.0

        if not steps[-1].done:
            reward, _ = self.Select_value(steps[-1].next_state, sess,self.vhct)
        states = []
        advantages = []
        value_targets = []
        actions = []
        for step in reversed(steps):

            reward = step.reward+self.discount_factor*reward
            advantage = reward-step.value
            states.append(step.state)
            actions.append(step.action)
            advantages.append(advantage)
            value_targets.append(reward)
        feed_dict = {
            self.local_policy.states: np.array(states),
            self.local_policy.advantages: advantages,
            self.local_policy.actions: actions,
            self.local_policy.ht: np.zeros((1, 256)),
            self.local_policy.ct: np.zeros((1, 256)),
            self.local_value.vstates: np.array(states),
            self.local_value.targets: value_targets,
            self.local_value.vht: np.zeros((1, 256)),
            self.local_value.vct: np.zeros((1, 256))

        }
        pnet_loss,vnet_loss,_,_ = sess.run([
            self.local_policy.loss,
            self.local_value.vloss,
            self.Policy_Train,
            self.Value_Train
        ], feed_dict)

        #self.vhct = [np.zeros((1, 256)), np.zeros((1, 256))]


        return pnet_loss,vnet_loss

