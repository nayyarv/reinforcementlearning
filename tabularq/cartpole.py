#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor

gymlogger.set_level(40)  # error only

import numpy as np
import random
from tabularq.tq import QDynamicTable, Epsilon


def show_video():
    import pygame
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
    env = wrap_env(gym.make('CartPoleCartPole-v1'))

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


alpha = 0.2
gamma = 0.99


class Agent:
    def __init__(self, env, bins=10):
        self.env = env
        self.Q = QDynamicTable(nA=2)
        self.epsilon = Epsilon(start=1.0, end=0.05, update_decrement=0.002)

        self.bins = [0, 0, 0, 0]
        self.bins[0] = np.array([-0.1, 0, 0.1])
        self.bins[1] = np.array([-0.75, 0, 0.75])
        thresh = np.pi / 180 * 12
        self.bins[2] = np.linspace(-thresh, thresh, 50)[1:-1]
        self.bins[3] = np.linspace(-1.7, 1.7, 9)[1:-1]  # drop endpoints

        for bn in self.bins:
            print(bn)

        # nbins = 5 * 5 * 32 * 7 = 5600

    def getAction(self, s):
        if self.epsilon.value > random.random():
            # print("Explore")
            # explore
            return self.env.action_space.sample()
        else:
            # print("   Exploit")
            # exploit
            return self.Q.get_max_a(s)

    def getSampleObs(self, n=100):
        # collect some observations to help populate bins
        for i in range(n):
            done = False
            yield self.env.reset()
            while not done:
                a = self.env.action_space.sample()
                s_1, reward, done, info = self.env.step(a)
                yield s_1

    def getSuggestedBins(self, bins=3):
        from matplotlib import pyplot as plt
        sample_obs = np.array(list(self.getSampleObs(1000)))
        for i, samp in enumerate(sample_obs.T):
            plt.subplot(220 + i + 1)
            plt.hist(samp, bins='auto')
        plt.show()
        percs = np.percentile(sample_obs, [2.5, 97.6], axis=0)
        # percs.T[2] = np.array([-12*np.pi/180, 12*np.pi/180])
        return [np.linspace(p[0], p[1], bins) for p in percs.T]

    def discretize(self, obs):
        """
        take observations and return a state
        """
        state = [int(np.digitize(obs[i], self.bins[i])) for i in range(len(self.bins))]
        # print(f"obs: {obs}¸ state: {state}")
        # return (0, 0, 0, 0)
        return tuple(state)

    def train(self, episodes=100, debug=False):
        self.epsilon.isTraining = True
        maxreward = 0
        rewardG = np.zeros(episodes)
        eps_mat = np.zeros(episodes)
        for i in range(episodes):
            if i % (episodes / 10) == 0:
                print(f"Episode: {i} of {episodes}, eps: {self.epsilon.value}")
            cumreward = 0
            done = False
            s = self.discretize(self.env.reset())
            if debug: print(f"Epsiode: {i}, start={s}, eps={self.epsilon.value}")
            while not done:
                a = self.getAction(s)
                s_1, reward, done, info = self.env.step(a)
                if done:
                    break
                    # reward = -.5
                if debug: print(s_1)
                s_1 = self.discretize(s_1)
                newq = self.Q.get_Q(s, a) + alpha * (reward + gamma * self.Q.get_max(s_1) - self.Q.get_Q(s, a))
                if debug: print(f"State: {s}, action={a}, newstate={s_1}, newq={newq:.2f}, eps={self.epsilon.value}")
                self.Q.set_Q(s, a, newq)
                # if debug: print(self.Q)
                s = s_1
                cumreward += reward
            rewardG[i] = cumreward
            eps_mat[i] = self.epsilon.value

            if cumreward > maxreward:
                print(f"New max {i}:{cumreward}")
                maxreward = cumreward

            self.epsilon.decrement(cumreward >= 80)
            # if self.epsilon.value == self.epsilon.end:
            #     print("Eps has reached minimum, ending early, adjust decrement logic")
            #     break
            # if reward == 1:
            #     self.epsilon.decrement()
            # else:
            #     self.epsilon.decrement(0.1)
        from matplotlib import pyplot as plt
        from scipy.signal import lfilter
        print(rewardG[:i:1000])
        dat = lfilter(np.ones(50)/50, 1, rewardG[:i])
        plt.style.use('ggplot')

        plt.plot(dat)
        plt.plot(eps_mat*max(dat))
        plt.title("Performance vs episode")
        plt.savefig("plot.png")
        plt.show()


    def run(self):
        self.env = wrap_env(self.env)
        self.epsilon.isTraining = False
        s = self.discretize(self.env.reset())
        steps = 0
        done = False
        print("Running agent")
        while not done:
            self.env.render()
            action = self.getAction(s)
            s_1_f, reward, done, info = self.env.step(action)
            s_1 = self.discretize(s_1_f)

            print(f"Done: {done}, Curr State: {s}, Action {['L', 'R'][action]}, New State: {s_1_f}")
            s = s_1
            steps += 1

        print("Episode finished after {} timesteps".format(steps))
        self.env.env.close()
        self.env.close()


def main():
    env = gym.make('CartPole-v1')
    # from gym.envs.classic_control import CartPoleEnv
    agent = Agent(env, 20)
    agent.train(episodes=20000, debug=False)
    # print(agent.Q)
    agent.run()
    # show_video()


if __name__ == '__main__':
    main()

import pytest


@pytest.fixture
def envT():
    return gym.make('CartPole-v0')


def test_digitize(envT):
    ag = Agent(envT)
    ag.bins = [np.linspace(-1, 1, 5), np.linspace(-1, 1, 5),
               np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)]
    print(ag.bins)
    # np.digitize()
    print(ag.discretize([-1, 1, 1, -1]))
