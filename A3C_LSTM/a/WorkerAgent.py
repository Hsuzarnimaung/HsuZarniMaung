import numpy as np
import tensorflow as tf
from Neural_Network import Create_Network
import cv2
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
            self.output_image = tf.image.crop_to_bounding_box(self.output_image, 34, 0, 160,160 )
            self.output_image = tf.image.resize(self.output_image, [160, 160], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output_image = tf.squeeze(self.output_image)
    def transfrom(self, state, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        return sess.run(self.output_image, {self.input_image: state})
def Make_Training_Operation(localnet, Global_net):

        Local_grads, _ = zip(*localnet.grads_and_vars)
        Local_grads, _ = tf.clip_by_global_norm(Local_grads, 5.0)
        _, Global_vars = zip(*Global_net.grads_and_vars)
        lgrads_g_vars = list(zip(Local_grads, Global_vars))
        Local_grads, _ = zip(*localnet.vgrads_and_vars)
        Local_grads, _ = tf.clip_by_global_norm(Local_grads, 5.0)
        _, Global_vars = zip(*Global_net.vgrads_and_vars)
        grads_g_vars = list(zip(Local_grads, Global_vars))


        return Global_net.optimizer.apply_gradients(lgrads_g_vars, global_step=tf.compat.v1.train.get_global_step()),Global_net.voptimizer.apply_gradients(grads_g_vars, global_step=tf.compat.v1.train.get_global_step())


def VMaking_Training_Operation(localnet,Global_net):
    Local_grads, _ = zip(*localnet.vgrads_and_vars)
    Local_grads, _ = tf.clip_by_global_norm(Local_grads, 5.0)
    _, Global_vars = zip(*Global_net.vgrads_and_vars)
    grads_g_vars = list(zip(Local_grads, Global_vars))
    return Global_net.voptimizer.apply_gradients(grads_g_vars, global_step=tf.compat.v1.train.get_global_step())
class Worker:
    def __init__(self, worker_name,
                 env,
                 global_policy,
                 returns_list,
                 steps_list,
                 global_counter,optimizer,
                 discount_factor=0.99,

                 max_global_steps=None):
        self.worker_name = worker_name
        self.env = env
        self.global_policy = global_policy

        self.global_counter = global_counter
        self.max_global_steps = max_global_steps

        self.discount_factor = discount_factor
        self.Preprocessing = Preprocesing_Input()
        self.steps_list=steps_list
        self.a=0
        self.optimizer=optimizer

        self.local_policy = Create_Network(global_policy.num_of_output,self.worker_name,self.optimizer)
        self.copy_params = Copy_Params_Scopes(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="Global"),
    tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.worker_name + "/")
          )
        self.Policy_Train = Make_Training_Operation(self.local_policy,self.global_policy)
        self.V_Train = VMaking_Training_Operation(self.local_policy, self.global_policy)
        self.state = None
        self.total_reward = 0.
        self.returns_list = returns_list
        self.step=0


    def get_env(self):
        return self.env
    def Run(self, sess, coord, t_max):
        with sess.as_default(), sess.graph.as_default():
            self.state = Stack_Frames(self.Preprocessing.transfrom(self.env.reset()))
            sess.run(self.copy_params)
            try:
                while not coord.should_stop():
                    sess.run(self.copy_params)
                    steps, global_step = self.Run_N_Steps(t_max, sess)
                    if self.max_global_steps is not None and global_step >= self.max_global_steps:
                        coord.request_stop()
                        return
                    self.Update_Network(steps, sess)
                    if(steps[-1].done):

                        self.step=0
                        self.local_policy.ht = np.zeros((1, 256))
                        self.local_policy.ct = np.zeros((1, 256))

                        #print(self.step)
            except tf.errors.CancelledError:
                return
    def Run_N_Steps(self, num, sess):
        steps = []
        for _ in range(num):
            action,value = self.Select_Action(self.state, sess)
            next_frame, reward, done, _ = self.env.step(action)
            next_state = Shift_Frames(self.state, self.Preprocessing.transfrom(next_frame))


            if done:
                print("Total reward:", self.total_reward, "Worker:", self.worker_name,"Step:",self.step)
                self.returns_list.append(self.total_reward)
                if len(self.returns_list) > 0 and len(self.returns_list) % 100 == 0:
                    print("**** Total average reward (last 100):", np.mean(self.returns_list[-100:]), "Total episode:", len(self.returns_list), "Count:", self.global_counter)
                self.total_reward = 0.
                self.steps_list.append(self.a)
            else:
                self.total_reward += reward

            step = Batch(self.state, action, reward, next_state, done, value)
            steps.append(step)
            global_step = next(self.global_counter)
            #cv2.imwrite(str(global_step) + ".png", next_state)

            self.a=global_step
            self.step+=1
            if done:
                self.state = Stack_Frames(self.Preprocessing.transfrom(self.env.reset()))
                break
            else:
                self.state = next_state
        return steps, global_step
    def Select_Action(self, state, sess):

        feed_dict = {self.local_policy.states: [state]}
        actions,vhat = sess.run([self.local_policy.sample_action,self.local_policy.vhat], feed_dict)
        #vhat=sess.run([],feed_dict)

        return actions[0],vhat[0]




    def Update_Network(self, steps, sess):

        reward = 0.0
        if not steps[-1].done:
            _,reward = self.Select_Action(steps[-1].next_state, sess)
            #reward = reward[0]

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
        #print(value_targets)
        feed_dict = {
                self.local_policy.states: states,
                self.local_policy.advantages: advantages,
                self.local_policy.actions: actions,
            self.local_policy.targets: value_targets


            }
        sess.run([
                self.local_policy.p_loss,
            self.local_policy.vloss,
            self.Policy_Train,


            ], feed_dict)



        #return pnet_loss

