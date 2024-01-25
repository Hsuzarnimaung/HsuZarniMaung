"""import gym
import random
env=gym.make("Breakout-v4",render_mode="human")
done=False
total_reward=0
env.reset()
while(done==False):

    action=env.action_space.n

    choice_action=random.randint(0,(action-1))
    _, reward, done,_,_ =env.step(choice_action)
    total_reward+=reward
    #print(reward, done)
print(total_reward)"""

import matplotlib.pyplot as plt
plt.figure(1)                # the first figure
plt.subplot(1)             # the first subplot in the first figure
plt.plot([1, 2, 3])
plt.subplot(2)             # the second subplot in the first figure
plt.plot([4, 5, 6])
plt.subplot(3)             # the second subplot in the first figure
plt.plot([4, 5, 6])


plt.figure(2)                # a second figure
plt.plot([4, 5, 6])          # creates a subplot() by default

plt.figure(1)                # first figure current;
                             # subplot(212) still current
plt.subplot(211)             # make subplot(211) in the first figure
                             # current
plt.title('Easy as 1, 2, 3')
plt.show()