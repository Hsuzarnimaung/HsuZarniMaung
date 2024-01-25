import threading

import tensorflow as tf
import gym
from Network import Network
from Worker import Worker
import itertools
def numofaction(env_name):
    env=gym.envs.make(env_name)
    action=env.action_space.n
    env.close()
    return action

tf.compat.v1.reset_default_graph()
with tf.device('/cpu:0'):
    Env_name="ALE/Pong-v5"
    action=numofaction(Env_name)
    Max_Global_Step=1e7
    numberofworkers=8
    with tf.compat.v1.variable_scope("Global"):
        global_policy,global_value=Network(action)
    global_counter=itertools.count()
    return_list=[]
    step_list=[]
    workers=[]
    for worker_id in range(8):
        worker=Worker(worker_name="worker_{}_{}".format(Env_name,worker_id),env_name=Env_name,
                      env=gym.envs.make(Env_name),
                      global_policy=global_policy,
                      global_value=global_value,
                      global_counter=global_counter,
                      return_list=return_list,
                      step_list=step_list,
                      discount_factor=0.99,max_global_step=Max_Global_Step)
        workers.append(worker)


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    coord=tf.train.Coordinator()
    worker_thread=[]
    for worker in workers:
        worker_fn=lambda:worker.Run(sess,coord,5)
        t=threading.Thread(target=worker_fn)
        t.start()
        worker_thread.append(t)
    coord.join(worker_thread, stop_grace_period_secs=350)