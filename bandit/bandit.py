#!/usr/bin/env python3
import numpy as np
import gym
import click

import gym_bandits  # necessary for gym.make

ZSCORE = 1.5


class Qtable:
    """
    Base class to record and update Q.
    Note I use nums=0 since I've used a slightly different formulation
    """

    def __init__(self, num=10):
        self.num_table = np.zeros(num)
        self.q = np.zeros(num)

    def update(self, action, reward):
        raise NotImplementedError

    def exploit(self):
        return np.argmax(self.q)

    def score(self):
        return np.dot(self.num_table, self.mu)


class QAverage(Qtable):
    def update(self, action, reward):
        self.q[action] = (self.q[action] * self.num_table[action] + reward) / (self.num_table[action] + 1)
        self.num_table[action] += 1
        self.mu = self.q


class QUCB(Qtable):
    """
    Uses q = mu + Z * se(mu)
    In this case se(mu) is just sqrt(sample variance). 
    Z is 1.96 for alpha=5%, but you can adapt to affect sureness. The greater Z the smaller the
    deviance from mu and thus the higher exploitation rate. Z = 0 is the same as QAverage
    """

    def __init__(self, num=10):
        self.mu = np.zeros(num)
        self.ss = np.zeros(num)
        super().__init__(num)

    def update(self, action, reward):
        self.mu[action] = (self.mu[action] * self.num_table[action] + reward) / (self.num_table[action] + 1)
        self.ss[action] = (self.ss[action] * self.num_table[action] + reward ** 2) / (self.num_table[action] + 1)
        self.num_table[action] += 1
        self.q[action] = self.mu[action] + ZSCORE * (np.sqrt(self.ss[action] - self.mu[action] ** 2))


# eps functions

def exponential(value, cst=20):
    return 1 - np.exp(-cst * value)


def invsq(value, cst=10):
    return 1 - 1 / (1 + cst * value ** 2)


def linear(value, _):
    return value


def const(value, cst=0.9):
    return cst


epsDict = {
    "exponential": exponential,
    "invsq": invsq,
    "linear": linear,
    "const": const
}


def percDecision(value):
    """exploitation vs exploration."""
    return np.random.random() < value


@click.command()
@click.option("--method", type=click.Choice(["Ave", "UCB"]), default="Ave")
@click.option("--bandit-seed", type=int, help="set bandit values seed")
@click.option("--seed", type=int, help="set exploration seed. Unset leaves it independent of banditseed")
@click.option("--numbandits", default=10)
@click.option("-N", default=1000)
@click.option("--eps-func", default="const", type=click.Choice(epsDict.keys()))  # overrides epsilon
@click.option("--eps-hp", type=float, help="If eps-func takes a parameter, this sets it. Look at code")
def main(method, bandit_seed, seed, numbandits, n, eps_func, eps_hp):
    N = n
    if bandit_seed:
        np.random.seed(bandit_seed)

    def eps(x):
        if eps_hp:
            # maybe a partial?
            f = lambda y: epsDict[eps_func](y, eps_hp)
        else:
            f = epsDict[eps_func]
        return f(x)

    env = gym.make('BanditTenArmedGaussian-v0')
    obs = env.reset()

    if seed:
        np.random.seed(seed)
    else:
        # reset randomness
        np.random.seed()

    explCount = 0
    if method == "Ave":
        q_table = QAverage(numbandits)
    else:
        q_table = QUCB(numbandits)

    for i in range(N):
        # continue
        if percDecision(eps(i / N)):
            action = q_table.exploit()
            explCount += 1
        else:
            # explore time
            action = np.random.randint(numbandits)

        _, reward, *_ = env.step(action)

        q_table.update(action, reward)

    actuals = [mu for mu, _ in env.env.r_dist]
    truebest = np.argmax(actuals)
    bestPerf = np.argmax(q_table.q)

    np.set_printoptions(precision=2)
    print(f"N = {N}, Qfunc: {method}, seed = {seed}")
    print("Num", q_table.num_table)
    print("Q  ", q_table.q)
    print("Mu ", q_table.mu)
    print(f"MAB  Id: {bestPerf}, Ave: {q_table.mu[bestPerf]:.2f}, ntimes: {q_table.num_table[bestPerf]}")
    print(f"True Id: {truebest}, Ave: {actuals[truebest]:.2f}")

    score = q_table.score() / (N * actuals[truebest])
    print(f"Score: {score:.2f}")
    print(f"nexploit: {explCount}, nexplore: {N-explCount}")


if __name__ == '__main__':
    main()
