#!/usr/bin/env python3

import random
import numpy as np
import gym
import gym_minigrid
import time

env = gym.make('MiniGrid-Dynamic-6x6-v0')

env.reset()
for i in range(0, 500):
    action = random.randint(0, env.action_space.n - 1)
    obs, reward, done, info = env.step(action)
    env.render('human')
    time.sleep(0.3)

    if done:
        env.reset()
