#  -*- coding: utf-8 -*-
from policy.policyIteration import value_iteration

__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np
from rlenvs.envs.gridworld import GridworldEnv


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random value function. Let's set endpoints to 0
    V = np.random.random(env.nS) * env.nS
    V[0] = 0
    V[-1] = 0

    deltaV = np.ones(env.nS)
    while np.sum(deltaV ** 2) > theta:
        for state in range(env.nS):
            v = 0
            for action, action_transitions in env.P[state].items():
                ps = 0
                # for each action taken, we have a bunch of possible state transitions outcomes
                # in this case we have a perfect model, so we have only 1 transition, but for posterity
                for transition, nextstate, reward, done in action_transitions:
                    ps += transition * (reward + discount_factor * V[nextstate])
                v += policy[state][action] * ps

            deltaV[state] = V[state] - v
            V[state] = v
        print(V, deltaV)
    return V


def main():
    # setup
    env = GridworldEnv()

    # policy evaluation
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v_valuation = policy_eval(random_policy, env)
    print(v_valuation)

    # policy value iteratio

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


def test_value():
    env = GridworldEnv()
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v = policy_eval(random_policy, env)
    expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)


