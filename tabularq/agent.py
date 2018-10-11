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
from tabularq.tq import QTable

import random


class Epsilon:
    def __init__(self, start=1.0, end=0.01, update_decrement=0.01):
        self.start = start
        self.end = end
        self.update_decrement = update_decrement
        self._value = self.start
        self.isTraining = True

    def decrement(self, count=1):
        self._value = max(self.end, self._value - self.update_decrement * count)
        return self

    @property
    def value(self):
        if self.isTraining:
            return self._value
        else:
            # always explore
            return 0

    @value.setter
    def value(self, val):
        self._value = val


alpha = 0.8
gamma = 0.9


class Agent:
    def __init__(self, map_name="1x8", is_slippery=False):
        self.env = FrozenLakeEnv(map_name=map_name, is_slippery=is_slippery)
        self.Q = QTable(num_actions=4)
        self.epsilon = Epsilon(start=1.0, end=0.01, update_decrement=0.01)

    def getAction(self, s):
        if self.epsilon.value > random.random():
            # print("Explore")
            # explore
            return self.env.action_space.sample()
        else:
            # print("   Exploit")
            # exploit
            return self.Q.get_max_a_for_Q(s)

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
        for i in range(episodes):
            done = False
            s = self.env.reset()
            print(f"Epsiode: {i}, start={s}, eps={self.epsilon.value}")
            # print(self.Q)
            while not done:
                a = self.getAction(s)
                s_1, reward, done, info = self.env.step(a)
                newq = self.Q.get_Q(s, a) + alpha * (reward + gamma * self.Q.get_max_Q(s_1) - self.Q.get_Q(s, a))
                # print(f"State: {s}, action={a}, newstate={s_1}, newq={newq:.2f}, reward={reward}")
                self.Q.set_Q(s, a, newq)
                # print(self.Q)
                s = s_1

            self.epsilon.decrement()

    def run(self):
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
            self.env.render()

        print("Episode finished after {} timesteps".format(steps))


if __name__ == '__main__':
    agent = Agent()
    agent.train(episodes=20)
    agent.run()
