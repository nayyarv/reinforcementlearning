#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np
from policy.policyvalue import policy_eval
from rlenvs.envs.gridworld import GridworldEnv


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    old_policy = np.zeros([env.nS, env.nA])
    policy = np.ones([env.nS, env.nA]) / env.nA

    while not np.allclose(old_policy, policy):
        # copy current policy
        old_policy[:] = policy[:]
        # calculate V for given policy
        V = policy_eval_fn(policy, env)

        for state in range(env.nS):
            # for each state, let's find all the next_states and choose the one with the highest value
            # since we have deterministic movement, we can just work out which action is best
            action_value_dict = {action: V[trans[0][1]] for action, trans in env.P[state].items()}
            # since there may be more than 1 optimal action, let's search and
            maxValue = max(action_value_dict.values())
            for action, value in action_value_dict.items():
                policy[state][action] = 1 if value == maxValue else 0
            # normalise to probability in case of more than one optimal action
            policy[state] /= np.sum(policy[state])

        # break
        print("iterating")

    return policy, V


def main():
    env = GridworldEnv()

    policy, v = policy_improvement(env)
    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    # print("Value Function:")
    # print(v)
    # print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")


if __name__ == '__main__':
    main()


def test_pol_iter():
    env = GridworldEnv()
    policy, v = policy_improvement(env)

    # Test the value function
    expected_v = np.array([0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)