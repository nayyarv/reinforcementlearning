#  -*- coding: utf-8 -*-

__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np
from rlenvs.envs.gridworld import GridworldEnv


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    # initialise V and policy
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA]) / env.nA
    exValue = np.zeros(env.nA)

    deltaV = np.ones(env.nS)
    while np.sum(deltaV ** 2) > theta:
        for state in range(env.nS):
            # let's work out the ex value of each action
            exValue[:] = 0
            for action, action_transitions in env.P[state].items():
                # for each action taken, we have a bunch of possible state transitions outcomes
                # in this case we have a perfect model, so we have only 1 transition, but for posterity
                for transition, nextstate, reward, done in action_transitions:
                    exValue[action] += transition * (reward + discount_factor * V[nextstate])

            # policy is action that best maximizes exValue
            maxVal = np.max(exValue)
            # not all max values are unique, find and normalise
            pol = (exValue == maxVal).astype(int)
            # print(exValue, pol)
            pol = pol / np.sum(pol)

            # update policy
            policy[state][:] = pol

            deltaV[state] = V[state] - maxVal
            V[state] = maxVal
        # print(V, deltaV)

    return policy, V


def main():
    # policy value iteration
    env = GridworldEnv()
    policy, v = value_iteration(env)

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Value Function:")
    print(v)
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")

    # Task 3, check policy
    print("Policy Probability Distribution:")
    print(policy)
    print("")


if __name__ == '__main__':
    main()


def test_val_iter():
    env = GridworldEnv()
    policy, v = value_iteration(env)
    # Test the value function
    expected_v = np.array([0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
