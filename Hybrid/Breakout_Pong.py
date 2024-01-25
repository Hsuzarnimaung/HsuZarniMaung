import gym

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

import threading
import multiprocessing
import pandas as pd
from nets import create_networks
from worker import Worker
from datetime import datetime
start=datetime.now()

ENV_NAME = "Breakout-v4"
MAX_GLOBAL_STEPS = 1e7
STEPS_PER_UPDATE = 20

def Env(ENV_NAME):
    return gym.envs.make(ENV_NAME)

env = Env(ENV_NAME)
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
NUM_WORKERS =4 # This one run with 12 processors

with tf.device("/cpu:0"):
    # Keeps track of number of updates we've performed
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Global policy and value nets
    with tf.variable_scope("global") as vs:
        net = create_networks(NUM_ACTIONS)

        # Global step iterator
    global_counter = itertools.count()
    global_counter2 = itertools.count()
    #epsodie_count=itertools.count()

    # Save returns
    returns_list = []
    step_list=[]

    # Create workers
    workers = []
    for worker_id in range(NUM_WORKERS):
        worker = Worker(
            name="worker_{}_{}".format(ENV_NAME,worker_id),
            env=Env(ENV_NAME),
            net=net,
            name2=ENV_NAME,
            step_list=step_list,
            global_counter=global_counter,
            returns_list=returns_list,
            discount_factor = 0.99,
            max_global_steps=MAX_GLOBAL_STEPS
            )
        workers.append(worker)
    returns_list2=[]
    step_list2=[]
    ENV_NAME="Pong-v4"
    for worker_id in range(NUM_WORKERS):
        worker = Worker(
            name="worker_{}_{}".format(ENV_NAME,worker_id),
            env=Env(ENV_NAME),
            net=net,
            name2=ENV_NAME,
            step_list=step_list2,
            global_counter=global_counter2,
            returns_list=returns_list2,
            discount_factor=0.99,
            max_global_steps=MAX_GLOBAL_STEPS
        )
        workers.append(worker)
    saver = tf.train.Saver(max_to_keep=5)
with tf.Session() as sess:
    if (True):
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state("Model_CNN_LSTM_batch_20/Breakout_Pong")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()

    # Start worker threads
    worker_threads = []
    for worker in workers:
        worker_fn = lambda: worker.run(sess, coord, STEPS_PER_UPDATE)
        t = threading.Thread(target=worker_fn)
        t.start()
        worker_threads.append(t)

    # wait for all workers to finish
    coord.join(worker_threads, stop_grace_period_secs=300)
    saver.save(sess, "./Model/Breakout_Pong_" + "/model" + str("_global_step") + ".cptk")
    end = datetime.now()
    print(f"Training time:{end - start}")
    df = pd.DataFrame.from_dict({"Rewards": returns_list,"Global_Step":step_list})
    df.to_excel("./Excel/Hybrid_Breakout_lstm.xlsx", index=False)
    df = pd.DataFrame.from_dict({"Rewards": returns_list2,"Global_Step":step_list2})
    df.to_excel("./Excel/Hybrid_Pong_lstm.xlsx", index=False)
    # Plot smoothed returns
    x = np.array(returns_list)
    y = smooth(x)
    plt.title("DemonAttack")
    plt.plot(x, label='orig')
    plt.plot(y, label='smoothed')
    plt.legend()
    plt.show()
    x = np.array(returns_list2)
    y = smooth(x)
    plt.title("SpaceInvaders")
    plt.plot(x, label='orig')
    plt.plot(y, label='smoothed')
    plt.legend()
    plt.show()