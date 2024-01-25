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
import sys
#global coord
coord= tf.train.Coordinator()
# Smooth the result of each episode
def smooth(x):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i + 1)].sum() / (i - start + 1))
    return y
def Env(ENV_NAME):
    return gym.envs.make(ENV_NAME)
ENV_NAME1="Breakout-v4"

# start the timer

#start = datetime.now()
    # Defining the first environment name, maximum global steps and batch size
    # create environment
env = Env(ENV_NAME1)

    # Get the number of action
NUM_ACTIONS = env.action_space.n
env.close()

    # Set number of workers
    # This one run with 4 processors
tf.compat.v1.reset_default_graph()
with tf.device("/cpu:0"):
    entropy = 0.01
    discount = 0.99
    worker_numbers = 4
    max_steps = 1500000
    STEPS_PER_UPDATE = 20
        # Global policy and value nets
    with tf.variable_scope("global"):
        net = create_networks(NUM_ACTIONS,reg=entropy)
        # Global step iterator for breakout agents
    global_counter = itertools.count()
        # Global step iterator for Pong agents
    global_counter2 = itertools.count()
        # epsodie_count=itertools.count()
        # Save returns and steps ...
    returns_list = []
    step_list = []

        # Create workers for breakout agent
    workers = []

    for worker_id in range(worker_numbers):
        worker = Worker(
        name="worker_{}_{}".format(ENV_NAME1, worker_id),
                env=gym.envs.make(ENV_NAME1),
                net=net,
                name2=ENV_NAME1,
                global_counter=global_counter,
                returns_list=returns_list,
                discount_factor=discount,
                max_global_steps=max_steps,
              entropy=entropy,
            )

        workers.append(worker)
        # Save returns and steps ...
    returns_list2=[]
    ENV_NAME2="Pong-v4"
    #global_counter2=itertools.count()
    for worker_id in range(worker_numbers):
        worker = Worker(
            name="worker_{}_{}".format(ENV_NAME2, worker_id),
            env=gym.envs.make(ENV_NAME2),
            net=net,
            name2=ENV_NAME2,
            global_counter=global_counter2,
            returns_list=returns_list2,
            discount_factor=discount,
            max_global_steps=max_steps,
            entropy=entropy,
        )

        workers.append(worker)
        # Saver for checkpoint of neural network
    saver = tf.train.Saver(max_to_keep=STEPS_PER_UPDATE)

with tf.Session() as sess:

    # to continue trained weights
    if (False):
        print('Loading Model of Breakout and Pong...')
        ckpt = tf.train.get_checkpoint_state("Model_CNN_LSTM_batch_20/Breakout_Pong")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    # Start worker threads
    worker_threads = []
    for worker in workers:
        worker_fn = lambda: worker.run(sess, coord, STEPS_PER_UPDATE)
        t = threading.Thread(target=worker_fn)
        t.start()
        worker_threads.append(t)
    start = datetime.now()
    # wait for all workers to finish
    coord.join(worker_threads, stop_grace_period_secs=300)

        # Saving the checkpoint after finishing global steps
    saver.save(sess, "./Model/"+ENV_NAME1+"_"+ENV_NAME2+"_After" + "/model" + str("_global_step") + ".cptk")
        # Ending the time
    end = datetime.now()
        # Calculating the training time
    print(f"Training time:{end - start}")
        # Saving the output from Breakout (such as rewards, global step, total of V(s),etc..)
    df = pd.DataFrame.from_dict(
            {"Rewards": returns_list})
    time_now = datetime.now()
    current_time = time_now.strftime("%H_%M_%S")
    df.to_excel(f"./Excel/Hybrid_Total_{ENV_NAME1}_{current_time}_CNN_LSTM.xlsx", index=False)
    df = pd.DataFrame.from_dict(
            {"Rewards": returns_list2})

    df.to_excel(f"./Excel/Hybrid_Total_{ENV_NAME2}_{current_time}_CNN_LSTM.xlsx", index=False)
    # Plot smoothed returns
    x = np.array(returns_list)
    y = smooth(x)
    plt.title(ENV_NAME1)
    plt.plot(x, label='orig')
    plt.plot(y, label='smoothed')
    plt.legend()

    plt.show()
    x = np.array(returns_list2)
    y = smooth(x)
    plt.title(ENV_NAME2)
    plt.plot(x, label='orig')
    plt.plot(y, label='smoothed')
    plt.legend()

    plt.show()

