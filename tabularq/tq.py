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


class QDynamicTable:
    def __init__(self, nA=4, nS=None):
        from collections import defaultdict
        self.num_actions = nA
        self.Q = defaultdict(lambda: [0] * nA)

    def __getitem__(self, item):
        return self.Q[item[0]][item[1]]

    def __setitem__(self, instance, value):
        self.Q[instance[0]][instance[1]] = value

    def get_Q(self, s, a):
        """Q(s, a): get the Q value of (s, a) pair"""
        return self.Q[s][a]

    def get_max(self, s):
        """max Q(s): get the max of all Q value of state s"""
        return max(self.Q[s])

    def set_Q(self, s, a, q):
        """Q(s, a) = q: update the q value of (s, a) pair"""
        self.Q[s][a] = q

    def get_max_a(self, s):
        """argmax_a Q(s, a): get the action which has the highest Q in state s"""

        mx = self.get_max(s)
        return random.choice([i for i, j in enumerate(self.Q[s]) if j == mx])

    def __str__(self):
        output = []
        for s in self.Q:
            output.append(s.__str__() + ": " + ["{:07.4f}".format(a) for a in self.Q[s]].__str__())
        output.sort()
        return "QTable (number of actions = " + str(self.num_actions) + ", states = " + str(
            len(output)) + "):\n" + "\n".join(output)


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


import pytest


@pytest.fixture(params=[QTable, QDynamicTable])
def Qt(request):
    return request.param(nA=4, nS=16)


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
