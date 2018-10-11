#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import random
import numpy as np


class QTable:
    def __init__(self, nS, nA):
        self.Q = np.zeros((nS, nA))

    def __getitem__(self, item):
        return self.Q[item]

    def __setitem__(self, instance, value):
        self.Q[instance] = value

    def get_Q(self, state, action):
        """Q(s, a): get the Q value of (s, a) pair"""
        return self.Q[state][action]

    def set_Q(self, state, action, q):
        """Q(s, a) = q: update the q value of (s, a) pair"""
        self.Q[state][action] = q

    def get_max(self, state):
        """max Q(s): get the max of all Q value of state s"""
        return np.max(self.Q[state])

    def get_max_a(self, state):
        """argmax_a Q(s, a): get the action which has the highest Q in state s"""

        mx = self.get_max(state)
        return random.choice(np.argwhere(self.Q[state] == mx))[0]

    def __str__(self):
        output = []
        nS, nA = self.Q.shape
        for s, state in enumerate(self.Q):
            output.append(f"{s:<4}: " + " ".join(f"{x:>6.3f}" for x in state))
        return f"QTable (number of actions {nA}, states = {nS}):\n" + '\n'.join(output)


import pytest


@pytest.fixture()
def Qt():
    return QTable(nA=4, nS=16)


def test_Qtable1(Qt):
    a = 1
    assert Qt[(11, a)] == 0, "Q value should be 0 to start with"


def test_Qtable2(Qt):
    a = 1
    s = 15
    Qt.set_Q(s, a, 90)
    assert Qt[(s, a)] == 90, "Updated Q value should equal 90"

    a = 2
    Qt.set_Q(s, a, 85)
    assert Qt.get_max(s) == 90, "Max Q should be 90"
    assert Qt.get_max_a(s) == 1, "Max action for state should be 1"

    print(Qt)
