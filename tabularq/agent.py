#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"
"""
Initialize Q(s, a), for all S, A, arbitrarily  (can do delayed initialisation)  
Repeat (for each episode)  
    Initialize S  
    Repeat (for each step of episode):  
        Choose A from S using policy derived from Q (epsilon - Greedy)  
        Take action, observe R, S'
        Q(S, A) <-- Q(S,A) + alpha [R + gamma * maxQ(S', a) - Q(S,A)]
        S <-- S'
    until S is terminal
"""

from tabularq.frozenlake import FrozenLakeEnv
from tabularq.tq import QTable, Epsilon

import random

alpha = 0.6
gamma = 0.7


class Agent:
    def __init__(self, map_name="1x8", is_slippery=False):
        self.env = FrozenLakeEnv(map_name=map_name, is_slippery=is_slippery)
        self.Q = QTable(self.env.nS, self.env.nA)
        self.epsilon = Epsilon(start=1.0, end=0.01, update_decrement=0.01)

    def getAction(self, s):
        if self.epsilon.value > random.random():
            # print("Explore")
            # explore
            return self.env.action_space.sample()
        else:
            # print("   Exploit")
            # exploit
            return self.Q.get_max_a(s)

    def train(self, episodes=20):
        """
        Initialize Q(s, a), for all S, A, arbitrarily  (can do delayed initialisation)
        Repeat (for each episode)
            Initialize S
            Repeat (for each step of episode):
                Choose A from S using policy derived from Q (epsilon - Greedy)
                Take action, observe R, S'
                Q(S, A) <-- Q(S,A) + alpha [R + gamma * maxQ(S', a) - Q(S,A)]
                S <-- S'
            until S is terminal
        """
        self.epsilon.isTraining = True
        reward = 0
        for i in range(episodes):
            done = False
            s = self.env.reset()
            print(f"Epsiode: {i}, start={s}, eps={self.epsilon.value}")
            while not done:
                a = self.getAction(s)
                # print(a)
                s_1, reward, done, info = self.env.step(a)
                newq = self.Q.get_Q(s, a) + alpha * (reward + gamma * self.Q.get_max(s_1) - self.Q.get_Q(s, a))
                # print(f"State: {s}, action={a}, newstate={s_1}, newq={newq:.2f}, reward={reward}")
                self.Q.set_Q(s, a, newq)
                # print(self.Q)
                s = s_1

            if reward == 1:
                self.epsilon.decrement()
            else:
                self.epsilon.decrement(0.1)

    def run(self, render=True):
        print("Running agent with this Q table")
        print("Q:", self.Q)
        self.epsilon.isTraining = False
        s = self.env.reset()
        steps = 0
        done = False
        self.env.render()
        while not done:
            action = self.getAction(s)
            # print(f"Action chosen = {action}")
            s_1, reward, done, info = self.env.step(action)
            s = s_1
            steps += 1
            if render: self.env.render()

        print("Episode finished after {} timesteps".format(steps))
        print("Success!" if reward == 1 else "Failure")


if __name__ == '__main__':
    # agent = Agent()
    # agent.train(episodes=20)
    # agent.run()

    agent = Agent(map_name="4x4", is_slippery=True)
    agent.train(episodes=1000)  # What is the good number of episodes to use? What if is_slipper is False?
    agent.run()

    # agent = Agent(map_name="8x8", is_slippery=False)
    # agent.train(episodes=1000)  # What is the good number of episodes to use? What if is_slipper is False?
    # agent.run(True)
