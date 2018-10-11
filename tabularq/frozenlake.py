#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"
from gym.envs.toy_text import FrozenLakeEnv as _Fzl

# retained for reference

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class FrozenLakeEnv(_Fzl):
    def __init__(self, desc=None, map_name="4x4", is_slippery=True):
        if map_name == "1x8":
            super().__init__("HFFSFFFG", map_name, is_slippery)
        else:
            super().__init__(desc, map_name, is_slippery)

    # def step(self, a):
    #     s_1, reward, done, info = super().step(a)
    #     if done and not reward:
    #         # fallen
    #         reward = -.5
    #     return s_1, reward, done, info
