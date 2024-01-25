import tensorflow as tf
from Network import Network
import numpy as np
class Preprocessing_Image:
    def __init__(self,env_name):
        with tf.compat.v1.variable_scope("Image_Tranformer"):
            self.input_image=tf.compat.v1.placeholder(shape=[210,160,3],dtype=tf.uint8)
            self.out_image=tf.image.rgb_to_grayscale(self.input_image)
            self.out_image=tf.image.crop_to_bounding_box(self.out_image,34,0,160,160)
            self.out_image=tf.image.resize(self.out_image,[84,84],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.out_image=tf.squeeze(self.out_image)
    def transform(self,state,sess=None):
        sess=sess or tf.compat.v1.get_default_session()
        return sess.run(self.out_image,{self.input_image:state})
class Batch:
    def __init__(self,state,action,reward,next_state,done):
        self.state=state
        self.action=action
        self.reward=reward
        self.next_state=next_state
        self.done=done
        self.value=0
def Making_Training(local_net,global_net):
    Lo_gar,_ =zip(*local_net.gar_and_var)
    Lo_gar,_=tf.clip_by_global_norm(Lo_gar,5.0)
    _,Glob_var=zip(*global_net.gar_and_var)
    Lo_gar_and_Glob_var=zip(Lo_gar,Glob_var)
    return global_net.trainer.apply_gradients(Lo_gar_and_Glob_var,global_step=tf.compat.v1.train.get_global_step())
def Copying_Parameters(sour_vars,dist_vars):
    src_var=list(sorted(sour_vars,key=lambda v:v.name))
    dst_var=list(sorted(dist_vars,key=lambda v:v.name))
    parameter=[]
    for sor, dst in zip(src_var,dst_var):
        op=dst.assign(sor)
        parameter.append(op)
    return parameter
def Stack_Frames(Frame):
    return np.stack([Frame]*4,axis=2)
def Shift_Frames(state,next_frame):
    return np.append(state[:,:,1:],np.expand_dims(next_frame,2),axis=2)
class Worker:
    def __init__(self,env_name,
                 worker_name,
                 env,
                 global_policy,
                 global_value,
                 return_list,
                 step_list,
                 global_counter,
                 discount_factor=0.99,
                 max_global_step=None):
        self.env_name=env_name
        self.worker_name=worker_name
        self.env=env
        self.global_policy=global_policy
        self.global_value=global_value
        self.return_list=return_list
        self.step_list=step_list
        self.global_counter=global_counter
        self.max_global_step=max_global_step
        self.discount_factor=discount_factor
        self.state=None
        self.total_reward=0.0
        self.Processing_image=Preprocessing_Image(self.env_name)
        with tf.compat.v1.variable_scope(self.worker_name):
            self.local_policy,self.local_value=Network(self.global_policy.num_of_action)
        self.Copy_Parameter=Copying_Parameters(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope="Global"),
                                               tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope=self.worker_name+"/"))

        self.Value_Train=Making_Training(self.local_value,self.global_value)

        self.Policy_Train = Making_Training(self.local_policy, self.global_policy)

    def Run(self,sess,coord,t_mas):
        with sess.as_default(),sess.graph.as_default():
            self.state=Stack_Frames(self.Processing_image.transform(self.env.reset()))
            try:
                while not coord.should_stop():
                    sess.run(self.Copy_Parameter)
                    steps, global_step=self.Run_n_step(sess,t_mas)
                    if self.max_global_step is not None and global_step>=self.max_global_step:
                        coord.request_stop()
                        return
                    self.Update_Network(sess,steps)
                    if steps[-1].done:
                        self.local_policy.ht=tf.zeros((1,256))
                        self.local_policy.ct=tf.zeros((1,256))
                        self.local_value.ht=tf.zeros((1,256))
                        self.local_value.ct=tf.zeros((1,256))

            except tf.errors.CancelledError:
                return

    def Run_n_step(self,sess,t_max):
        steps=[]
        for _ in range(t_max):
            action=self.Selection_Action(self.state,sess)
            next_frame,reward,done,_=self.env.step(action)
            next_state=Shift_Frames(self.state,self.Processing_image.transform(next_frame))
            if done:
                print("Total Reward:",self.total_reward,"Worker:",self.worker_name)
                self.return_list.append(self.total_reward)
                if len(self.return_list)>0 and len(self.return_list)%100==0:
                    print("*****Total average Reward of the last 100 episode:",np.mean(self.return_list[-100:]),"Total Episode:",len(self.return_list),"Global Counter:",self.global_counter)
                self.total_reward=0
            else:
                self.total_reward+=reward
            step = Batch(self.state,action,reward,next_state,done)
            steps.append(step)
            global_step=next(self.global_counter)
            if done:
                self.state=Stack_Frames(self.Processing_image.transform(self.env.reset()))
                break
            else:
                self.state=next_state
        return steps,global_step

    def Selection_Action(self, state, sess):
        feed_dist = {self.local_policy.states: [state]}
        action = sess.run(self.local_policy.sample_action, feed_dist)
        return action[0]

    def Selection_Value(self, state, sess):
        feed_dist = {self.local_value.states: [state]}
        vhat = sess.run(self.local_value.vhat, feed_dist)
        return vhat[0]

    def Update_Network(self,sess,steps):
        reward=0.0
        if not steps[-1].done:
            reward=self.Selection_Value(steps[-1].next_state,sess)
        states=[]
        advantages=[]
        value_targets=[]
        actions=[]
        for step in reversed(steps):
            reward = step.reward + self.discount_factor * reward
            advantage = reward - self.Selection_Value(step.state, sess)

            # Accumulate updates
            states.append(step.state)
            actions.append(step.action)
            advantages.append(advantage)
            value_targets.append(reward)
        feed_dist={
            self.local_policy.states:np.array(states),
            self.local_policy.advantage:advantages,
            self.local_policy.actions:actions,
            self.local_value.states:np.array(states),
            self.local_value.targets:value_targets,
        }
        sess.run([

                            self.local_policy.loss,
                          self.local_value.loss,
                          self.Policy_Train,
                          self.Value_Train],feed_dist)

