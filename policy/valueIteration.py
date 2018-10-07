#  -*- coding: utf-8 -*-

__author__ = "Varun Nayyar <nayyarv@gmail.com>"


import numpy as np
from rlenvs.envs.gridworld import GridworldEnv
from .policyvalue import policy_eval


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

    V = np.zeros(env.nS)

    # Task 3 (or 2.5) - Ignore policy for task 2
    policy = np.zeros([env.nS, env.nA])

    # Implement!
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


