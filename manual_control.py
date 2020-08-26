#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

def redraw(img):
    img = env.render('rgb_array', tile_size=args.tile_size)
    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    window_2.show_img(obs)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == 't':
        step(env.actions.toggle)
        return
    if event.key == 'p':
        step(env.actions.pickup)
        return
    if event.key == 'd':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-Dynamic-6x6-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=True,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

args = parser.parse_args()

env = gym.make(args.env)

# if args.agent_view:
#     env = RGBImgPartialObsWrapper(env)
#     env = ImgObsWrapper(env)

window_2 = Window('obs gym_minigrid - ' + args.env)
window = Window('gym_minigrid - ' + args.env)

window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window_2.show(block=True)
window.show(block=True)
