import itertools
from time import sleep
import tensorflow as tf
import gym
from Neural_Network import Create_Policy_Network,Create_Value_Network
from WorkerAgent import Worker
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
start=datetime.now()
import threading
#create Environment
def Environment(Env_name):
    return gym.envs.make(Env_name)
#Get the number of Action
def Num_Of_Action(Env):

    action=Env.action_space.n
    Env.close()
    return action
#Create Global NetWork
def Global_NetWork(num_actions):

    policy_net = Create_Policy_Network(num_actions)
    value_net = Create_Value_Network()
    return policy_net, value_net

#Create Workers
def Worker_Agents(Num_Workers, Env_name, Global_policy, Global_value, MAX_Global_Step, Model_Path, ReturnList,Iterator,steps_list):
    Agents = []

    for workerId in range(Num_Workers):
        agent = Worker(worker_name="Worker_{}_{}".format(Env_name, workerId),
                     env=Environment(Env_name), global_policy=Global_policy,
                     global_value=Global_value, returns_list=ReturnList, global_counter=Iterator,steps_list=steps_list,model_path=Model_Path, discount_factor=0.99,
                       max_global_steps=MAX_Global_Step
                     )

        Agents.append(agent)
    return Agents
#Training on only cpu threads
#Dividing Sessions and starting threading
def Create_Sessions(Agents,Step):
    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        coops = tf.train.Coordinator()
        Threads = []
        for Agent in Agents:
            worker = lambda: Agent.Run(session, coops, Step)
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
        plt.plot(x, label='orig')
        plt.plot(y, label='smoothed')
        plt.legend()
        plt.show()
def smooth(x):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i+1)].sum() / (i - start + 1))
    return y

with tf.device("/cpu:0"):
    #global_step = tf.Variable(1, name="global_step")
    Env_name = "Breakout-v4"
    env = Environment(Env_name)
    actions = Num_Of_Action(env)
    num_of_worker = 8
    with tf.compat.v1.variable_scope("Global") as vs:
        global_policy, global_value = Global_NetWork(actions)
    Max_global_step = 1e7
    Model_Path = ''
    ReturnList = []
    steps_list=[]
    Agents=[]
    Iterator = itertools.count()
    for workerId in range(8):
        agent = Worker(worker_name="Worker_{}_{}".format(Env_name, workerId),
                       env=Environment(Env_name), global_policy=global_policy,
                       global_value=global_value, returns_list=ReturnList, global_counter=Iterator,
                       steps_list=steps_list, model_path=Model_Path, discount_factor=0.99,
                       max_global_steps=Max_global_step
                       )

        Agents.append(agent)
   # Agents = Worker_Agents(num_of_worker, Env_name, global_policy, global_value, Max_global_step, Model_Path, ReturnList,Iterator,steps_list)

Create_Sessions(Agents,5)


