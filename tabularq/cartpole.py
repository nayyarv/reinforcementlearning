#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor

gymlogger.set_level(40)  # error only

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import math
import pygame


def show_video():
    import glob
    from moviepy.editor import VideoFileClip

    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        clip = VideoFileClip(mp4)
        clip.close()
    else:
        print("Could not find video")


def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env


def sample():
    env = wrap_env(gym.make('CartPole-v0'))

    observations = []
    for i_episode in range(1):
        observation = env.reset()
        observations.append(observation)
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            # record state, reward, done for printing
            observations.append((observation, reward, done))
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.env.close()
    env.close()
    for ob in observations: print(ob)
    show_video()
