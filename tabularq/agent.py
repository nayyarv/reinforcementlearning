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

class Epsilon(object):
    def __init__(self, start=1.0, end=0.01, update_decrement=0.01):
        self.start = start
        self.end = end
        self.update_decrement = update_decrement
        self._value = self.start
        self.isTraining = True

    def decrement(self, count=1):
        self._value = max(self.end, self._value - self.update_decrement * count)
        return self

    def value(self):
        if not self.isTraining:
            return 0.0
        else:
            return self._value


class Agent():
    def __init__(self):
        self.env = FrozenLakeEnv(map_name="1x8", is_slippery=False)
        self.Q = QTable(num_actions=4)
        self.epsilon = Epsilon(start=1.0, end=0.01, update_decrement=0.01)

    def getAction(self, s):
        if self.epsilon.value() > random.random:
            # explore
            return self.env.action_space.sample()
        else:
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
        # TODO:
        # - get training loop working with random action
        # - store and update Q values
        self.epsilon.isTraining = True
        self.env.reset()
        for i in range(episodes):
            a = self.getAction()
            s_1, reward, done, info = self.env.step(a)


        pass

    def run(self):
        print("Running agent with this Q table")
        print("Q:", self.Q)
        self.epsilon.isTraining = False
        s = self.env.reset()
        s = tuple([s])
        print(s)
        steps = 0
        while True:
            self.env.render()
            action = self.getAction(s)
            s_1, reward, done, info = self.env.step(action)
            s_1 = tuple([s_1])
            s = s_1
            steps += 1
            if done:
                print("Episode finished after {} timesteps".format(steps))
                break


if __name__ == '__main__':
    agent = Agent()
    agent.train(episodes=5)
    agent.run()
