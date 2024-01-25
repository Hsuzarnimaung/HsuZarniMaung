import itertools
from time import sleep
import tensorflow as tf
import gym
from Neural_Network import Create_Network
from WorkerAgent import Worker
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
start=datetime.now()
import threading
from threading import Thread, Lock
RANDOM_SEED = 1234
#create Environment
def Environment(Env_name):
    return gym.envs.make(Env_name)
#Get the number of Action
def Num_Of_Action(Env):

    action=Env.action_space.n
    Env.close()
    return action
#Create Global NetWork


#Create Workers

#Training on only cpu threads
#Dividing Sessions and starting threading


def smooth(x):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i+1)].sum() / (i - start + 1))
    return y
global global_policy
with tf.device("/cpu:0"):
    Env_name = "ALE/Breakout-v5"
    env = Environment(Env_name)
    actions = Num_Of_Action(env)
    num_of_worker = 8
    with tf.compat.v1.variable_scope("Global"):
        global_policy= Create_Network(actions)

    Max_global_step = 2e7
    Model_Path = ''
    ReturnList = []
    steps_list=[]
    Agents=[]
    Iterator = itertools.count()


    for workerId in range(8):
        agent = Worker(worker_name="Worker_{}_{}".format(Env_name, workerId),
                       env=Environment(Env_name), global_policy=global_policy,
                        returns_list=ReturnList, global_counter=Iterator,
                       steps_list=steps_list, discount_factor=0.99,
                       max_global_steps=Max_global_step,lock=Lock()
                       )

        Agents.append(agent)

with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        coops = tf.train.Coordinator()
        Threads = []
        for Agent in Agents:
            worker = lambda: Agent.Run(session, coops, 5)
            thread = threading.Thread(target=worker)
            thread.start()
            #sleep(0.5)
            Threads.append(thread)
        coops.join(Threads, stop_grace_period_secs=350)
        df = pd.DataFrame.from_dict({"Rewards": ReturnList, "Global Steps:": steps_list})
        df.to_excel("SpaceInvaders_Lstm.xlsx", index=False)
        end = datetime.now()
        print(f"Training time:{end - start}")
        # Plot smoothed returns
        x = np.array(ReturnList)
        y = smooth(x)
        plt.title("SpaceInvaders")
        plt.plot(x, label='orig')
        plt.plot(y, label='smoothed')
        plt.legend()
        plt.show()




