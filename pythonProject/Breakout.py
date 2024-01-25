import gym #initilize gym
import random
env=gym.make("Breakout-v4",render_mode="human") #build breakout game environment from gym
done=False # for condition
total_rewards=0 #total rewards for each game
env.reset()
while(done==False):
    action=env.action_space.n #get the number of total action
    choice_action=random.randint(0,action-1) #choosing action randomly
    _,reward,done,_,_=env.step(choice_action) #doing action in environment
    total_rewards+=reward #collection rewards
print(total_rewards)