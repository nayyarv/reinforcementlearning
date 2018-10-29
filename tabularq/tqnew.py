"""
Rewritten version of tabular q to be more elegant and pythonic
"""

import random
import numpy as np
import gym

from matplotlib import pyplot as plt

plt.style.use("ggplot")


class QTable:
    ALPHA = 0.1

    def __init__(self, statedim, num_actions):
        from collections import defaultdict
        self.qdict = defaultdict(lambda: [0] * num_actions)

        # manually specify bins after inspection
        self.bins = [0, 0, 0, 0]
        self.bins[0] = np.array([-0.1, 0, 0.1])
        self.bins[1] = np.array([-0.75, 0, 0.75])
        thresh = np.pi / 180 * 12
        self.bins[2] = np.linspace(-thresh, thresh, 50)[1:-1]
        self.bins[3] = np.linspace(-1.7, 1.7, 9)[1:-1]

    def _discretize(self, obs):
        """assume a tuple is already discretized"""
        if not isinstance(obs, tuple):
            state = [int(np.digitize(obs[i], self.bins[i])) for i in range(len(self.bins))]
            return tuple(state)
        return obs

    def __getitem__(self, item):
        try:
            state, action = item
        except ValueError:
            # can't unpack, assume we've only been passed in a state
            return self.qdict[self._discretize(item)]
        else:
            return self.qdict[self._discretize(state)][action]

    def __setitem__(self, key, value):
        """
        Only allowed to set a state, value pair.
        Update is as per alpha value
        """
        try:
            state, action = key
        except ValueError:
            raise ValueError("can only set item on (state, action) pair")
        else:
            state = self._discretize(state)
            self.qdict[state][action] = (self.qdict[state][action] * (1 - QTable.ALPHA)
                                         + value * QTable.ALPHA)

    def get_max(self, state):
        """get maximum q score for a state over actions"""
        state = self._discretize(state)
        return max(self.qdict[state])

    def get_arg_max(self, state):
        """which action gives max q score. If identical max q scrores exist,
           this'll return a choice
        """
        state = self._discretize(state)
        mx = self.get_max(state)
        return random.choice([i for i, j in enumerate(self.qdict[state]) if j == mx])


def exploit(epsilon):
    """exploitation increases as epsilon decreases"""
    return random.random() > epsilon


def ma(signal, window_size):
    from scipy.signal import lfilter
    filt = np.ones(window_size) / window_size
    a = np.array([1])
    return lfilter(filt, a, signal)


GAMMA = 0.9


class Agent:

    def __init__(self, Qstate, gamma=GAMMA):
        self.env = gym.make('CartPole-v0')
        self.gamma = GAMMA
        self.Qstate = Qstate(statedim=self.env.observation_space.shape[0],
                             num_actions=self.env.action_space.n)

    def get_action(self, state, epsilon):
        if exploit(epsilon):
            return self.Qstate.get_arg_max(state)
        else:
            return self.env.action_space.sample()

    def train(self, num_epsiodes, initeps=1, finaleps=0.05):
        epsdecay = (initeps - finaleps) / num_epsiodes
        epsilon = initeps

        test_window = int(num_epsiodes / 20)

        num_steps = np.zeros(num_epsiodes)
        eps_vals = np.zeros(num_epsiodes)

        for i in range(num_epsiodes):
            state = self.env.reset()
            steps = 0
            epsilon -= epsdecay

            done = False

            while not done:
                action = self.get_action(state, epsilon=epsilon)
                new_state, reward, done, info = self.env.step(action)
                if not done:
                    # add the future reward * decay
                    reward += self.gamma * self.Qstate.get_max(new_state)
                    steps += 1
                else:
                    pass
                    # print(f"Done: {new_state}, {steps}")

                self.Qstate[state, action] = reward
                state = new_state

            num_steps[i] = steps
            eps_vals[i] = epsilon

            if i % test_window == 0:
                # every 5%
                upp = i // test_window * test_window
                low = upp - test_window
                print(f"{i}: eps:{epsilon:.2f}, {steps}, {low}, {upp} ave: "
                      f"{np.sum(num_steps[low:upp])/test_window}")

        return num_steps, eps_vals

    def run(self, render=True):
        done = False
        steps = 0
        state = self.env.reset()
        while not done:
            if render: self.env.render()
            action = self.get_action(state, epsilon=0)
            new_state, reward, done, info = self.env.step(action)
            state = new_state
            steps += 1

        print(f"Numsteps: {steps}")
        self.env.close()


def to_row(array):
    if array.ndim == 1:
        return array.reshape(1, len(array))
    return array


HIDDEN_NODES = 20


class CartQCustom:
    def __init__(self, state_dim, action_dim, hidden_nodes=HIDDEN_NODES):
        from keras import Model
        from keras.layers import Dense, Input, Dot
        self.state_dim = state_dim
        self.action_dim = action_dim

        state = Input((state_dim,))
        h1 = Dense(hidden_nodes, activation="relu")(state)
        h2 = Dense(hidden_nodes, activation="relu")(h1)
        qvals = Dense(action_dim, activation="linear")(h2)
        # this is the qval model, however we're going to add a mask to train
        # on. This submodel will be trained too and can be used later
        self.qvalmod = Model(inputs=state, outputs=qvals)

        # now mask with chosen action
        action_in = Input((action_dim,))
        # the dot product with axis=1 will give us what we need
        max_sel = Dot(1, name='max_sel')([qvals, action_in])
        # input is state and action chosen, output is q-val for given actionn

        # we build the model to train with
        model = Model(inputs=[state, action_in], outputs=max_sel)
        model.compile("adam", loss='mean_squared_error')
        self.model = model

    def fit_single(self, state, action, output):
        """
        Args:
            state (np.array): (self.statedim)
            action (int):
            output (float):

        """
        state = to_row(state)
        a = np.zeros(self.action_dim)
        a[action] = 1
        action = to_row(a)

        output = np.array([output])

        self.model.fit([state, action], output, verbose=False)

    def pred_qval(self, state):
        return self.qvalmod.predict(to_row(np.array(state)))

    def __getitem__(self, item):
        try:
            state, action = item
        except ValueError:
            # can't unpack, assume we've only been passed in a state
            return self.pred_qval(item)
        else:
            return self.pred_qval(state)[action]

    def __setitem__(self, key, value):
        """
        Only allowed to set a state, value pair.
        Update is as per alpha value
        """
        try:
            state, action = key
        except ValueError:
            raise ValueError("can only set item on (state, action) pair")
        else:
            self.fit_single(state, action, value)

    def get_max(self, state):
        """get maximum q score for a state over actions"""
        qvals = self[state]
        return max(qvals)

    def get_arg_max(self, state):
        """which action gives max q score. If identical max q scrores exist,
           this'll return a choice
        """
        qvals = self[state]
        mx = max(qvals)
        return random.choice([i for i, j in enumerate(qvals) if j == mx])


if __name__ == '__main__':
    ag = Agent(QTable)
    NTRAIN = 50000

    ns, ev = ag.train(NTRAIN)
    smoothed = ma(ns, NTRAIN // 20)
    plt.plot(smoothed)
    # plt.plot(ev*max(smoothed))
    plt.show()
    ag.run()
