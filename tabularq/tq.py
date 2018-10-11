#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

# from tabularq.frozenlake import *

import random

import collections


class QTable():
    def __init__(self, num_actions=4):
        self.num_actions = num_actions
        self.Q = collections.defaultdict(lambda: [0] * num_actions)

    def get_Q(self, s, a):
        """Q(s, a): get the Q value of (s, a) pair"""
        return self.Q[s][a]

    def get_max_Q(self, s):
        """max Q(s): get the max of all Q value of state s"""
        return max(self.Q[s])

    def set_Q(self, s, a, q):
        """Q(s, a) = q: update the q value of (s, a) pair"""
        self.Q[s][a] = q

    def get_max_a_for_Q(self, s):
        """argmax_a Q(s, a): get the action which has the highest Q in state s"""

        mx = self.get_max_Q(s)
        return random.choice([i for i, j in enumerate(self.Q[s]) if j == mx])

    def __str__(self):
        output = []
        for s in self.Q:
            output.append(s.__str__() + ": " + ["{:07.4f}".format(a) for a in self.Q[s]].__str__())
        output.sort()
        return "QTable (number of actions = " + str(self.num_actions) + ", states = " + str(
            len(output)) + "):\n" + "\n".join(output)


import pytest


@pytest.fixture()
def Qt():
    return QTable(num_actions=4)


def test_Qtable1(Qt):
    s = tuple([5, 6])
    a = 1
    assert Qt.get_Q(s, a) == 0, "Q value should be 0 to start with"


def test_Qtable2(Qt):
    a = 1
    s = tuple([5, 3])
    Qt.set_Q(s, a, 90)
    assert Qt.get_Q(s, a) == 90, "Updated Q value should equal 90"

    a = 2
    Qt.set_Q(s, a, 85)
    assert Qt.get_max_Q(s) == 90, "Max Q should be 90"
    assert Qt.get_max_a_for_Q(s) == 1, "Max action for state should be 1"

    print(Qt)
