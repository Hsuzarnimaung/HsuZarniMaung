import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import threading
import pandas as pd

from nets import create_networks
from worker import Worker
from datetime import datetime
start=datetime.now()
ENV_NAME = "Riverraid-v4"
MAX_GLOBAL_STEPS = 1e7
STEPS_PER_UPDATE = 5

def Env():
    return gym.envs.make(ENV_NAME)

# Depending on which game you choose, we may need to limit the action space (cut out Unnecessary options from gym)
if ENV_NAME == "Pong-v4" or ENV_NAME == "Breakout-v4":
    NUM_ACTIONS = 4
else:
    env = Env()
    NUM_ACTIONS = env.action_space.n
    env.close()

def smooth(x):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i+1)].sum() / (i - start + 1))
    return y

# Set number of workers
NUM_WORKERS = 4 # This one run with 12 processors

with tf.device("/cpu:0"):
    #saver = tf.train.Saver()
    # Keeps track of number of updates we've performed
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Global policy and value nets
    with tf.variable_scope("global") as vs:
        policy_net, value_net = create_networks(NUM_ACTIONS)

        # Global step iterator
    global_counter = itertools.count()
    global_counter2=itertools.count()

    # Save returns
    returns_list = []
    steps_list1=[]
    # Create workers
    workers = []
    for worker_id in range(NUM_WORKERS):
        worker = Worker(
            name="worker_{}_{}".format(ENV_NAME,worker_id),
            env=Env(),
            policy_net=policy_net,
            value_net=value_net,
            global_counter=global_counter,
            returns_list=returns_list,
            Steps_list=steps_list1,
            discount_factor = 0.99,
            max_global_steps=MAX_GLOBAL_STEPS
            )
        workers.append(worker)
    ENV_NAME = "DemonAttack-v4"
    returns_list2=[]
    steps_list2=[]
    for worker_id in range(NUM_WORKERS):
        worker = Worker(
            name="worker_{}_{}".format(ENV_NAME,worker_id),
            env=Env(),
            policy_net=policy_net,
            value_net=value_net,
            global_counter=global_counter,
            returns_list=returns_list2,
            Steps_list=steps_list2,
            discount_factor = 0.99,
            max_global_steps=MAX_GLOBAL_STEPS
            )
        workers.append(worker)

    saver=tf.train.Saver(max_to_keep=STEPS_PER_UPDATE)

with tf.Session() as sess:
    if (False):
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state("SpaceInvaders_DemonAttack")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()


    # Start worker threads
    worker_threads = []

    for worker in workers:
        worker_fn = lambda: worker.run(sess, coord, STEPS_PER_UPDATE,saver)
        t = threading.Thread(target=worker_fn)
        t.start()
        worker_threads.append(t)


    # wait for all workers to finish
    coord.join(worker_threads, stop_grace_period_secs=300)
    saver.save(sess,"SpaceInvaders_DemonAttack/model-"+str(MAX_GLOBAL_STEPS)+".cptk`")
    df=pd.DataFrame.from_dict({"Rewards":returns_list,"Global Step":steps_list1})
    df.to_excel("Hybrid_SpaceInvaders_lstm.xlsx",index=False)
    df=pd.DataFrame.from_dict({"Rewards":returns_list2,"Global Step":steps_list2})
    df.to_excel("Hybrid_DemonAttack_lstm.xlsx",index=False)
    end = datetime.now()
    print(f"Training time:{end - start}")
    # Plot smoothed returns
    x = np.array(returns_list)
    y = smooth(x)
    plt.title("Breakout")
    plt.plot(x, label='orig')
    plt.plot(y, label='smoothed')
    plt.legend()
    plt.show()
    x = np.array(returns_list2)
    y = smooth(x)
    plt.title("Pong")
    plt.plot(x, label='orig')
    plt.plot(y, label='smoothed')
    plt.legend()
    plt.show()