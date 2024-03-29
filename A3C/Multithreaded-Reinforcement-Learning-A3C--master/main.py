import gym
import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import shutil
import threading
import multiprocessing
from datetime import datetime
import pandas as pd
start=datetime.now()
from nets import create_networks
from worker import Worker

ENV_NAME = "Breakout-v0"
MAX_GLOBAL_STEPS = 1e3
STEPS_PER_UPDATE = 5

def Env():
    return gym.envs.make(ENV_NAME)

# Depending on which game you choose, we may need to limit the action space (cut out Unnecessary options from gym)
if ENV_NAME == "Pong-v0" or ENV_NAME == "Breakout-v0":
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
NUM_WORKERS = 1#multiprocessing.cpu_count() # This one run with 12 processors

with tf.device("/cpu:0"):
    # Keeps track of number of updates we've performed
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Global policy and value nets
    with tf.variable_scope("global") as vs:
        policy_net, value_net = create_networks(NUM_ACTIONS)

        # Global step iterator
    global_counter = itertools.count()

    # Save returns
    returns_list = []

    # Create workers
    workers = []
    steps_list=[]
    for worker_id in range(NUM_WORKERS):

        worker = Worker(
            name="worker_{}".format(worker_id),
            env=Env(),
            policy_net=policy_net,
            value_net=value_net,
            global_counter=global_counter,
            returns_list=returns_list,
            steps_list=steps_list,
            discount_factor = 0.99,
            max_global_steps=MAX_GLOBAL_STEPS
            )
        workers.append(worker)
    saver=tf.train.Saver(max_to_keep=5)
with tf.Session() as sess:
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
    df = pd.DataFrame.from_dict({"Rewards": returns_list, "Global Steps:": steps_list})
    df.to_excel("Pong_Lstm.xlsx", index=False)
    end = datetime.now()
    print(f"Training time:{end - start}")
    # Plot smoothed returns
    x = np.array(returns_list)
    y = smooth(x)
    plt.title(ENV_NAME)
    plt.plot(x, label='orig')
    plt.plot(y, label='smoothed')
    plt.legend()
    plt.show()