import gym
import random
env=gym.make("SpaceInvaders-v4",render_mode="human")
episode=5
for ep in range(episode):
    state=env.reset()
    done=False
    score=0
    while not done:
        env.render(mode="rgb_array")
        action=random.choice([0,1,2,3,4,5])
        n_state,reward,done,infor=env.step(action)
        score+=reward
        print("episode{},Score{}".format(ep,score))
env.close()