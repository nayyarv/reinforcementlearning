import gym
import tensorflow as tf
import numpy as np
import random

GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.6  # starting value of epsilon
FINAL_EPSILON = 0.05  # final value of epsilon
EPSILON_DECAY_STEPS = 1000
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy
HIDDEN_NODES = 20
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

ACTION_DICT = [np.array([1, 0]), np.array([0, 1])]


class CartQ:
    def __init__(self, dimin=4, dimout=2):
        from keras import Sequential

        from keras.layers import Dense, Maximum

        model = Sequential()
        # add 2 dense hidden layers. First layer must have input size
        model.add(Dense(HIDDEN_NODES, activation='relu', input_dim=dimin))
        model.add(Dense(HIDDEN_NODES, activation='relu'))
        # combine the 2nd hidden layer into action nodes
        model.add(Dense(dimout, activation='linear'))
        # convert into a maximum so we know which action to take.
        model.compile('adam', loss='mean_squared_error')

        self.model = model

def to_row(array):
    if array.ndim == 1:
        return array.reshape(1, len(array))
    return array


def action_to_mask(action, action_space):
    msk = np.zeros((len(action), action_space))
    mdict = []
    for i in range(action_space):
        submask = np.zeros(action_space)
        submask[i] = 1
        mdict[i] = submask

    for i in range(action_space):
        msk[action==i] = mdict[i]
    return msk



class CartQCustom:
    def __init__(self, state_dim, action_dim):
        from keras import Model
        from keras.layers import Dense, Input, Dot
        self.state_dim = state_dim
        self.action_dim = action_dim

        state = Input((state_dim,))
        h1 = Dense(HIDDEN_NODES, activation="relu")(state)
        h2 = Dense(HIDDEN_NODES, activation="relu")(h1)
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

        self.model.fit([state, action], output)

    def pred_qval(self, state):
        return self.qvalmod.predict(to_row(state))


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.Q = CartQCustom(env.observation_space.shape[0], env.action_space.n)

    def get_action(self, state, epsilon):

        if epsilon > random.random():
            # exploit
            qvals = self.Q.pred_qval(state)
            return np.argmax(qvals)
        else:
            # expore
            return self.env.action_space.sample()

    def train(self, num_episodes):
        epsilon = INITIAL_EPSILON






if __name__ == '__main__':
    Q = CartQCustom(env.observation_space.shape[0], env.action_space.n)
    state = np.random.random((100, 4))
    rds = np.random.random(100)
    ls = []
    for i in range(100):
        if rds[i] > 0.5:
            ls.append(ACTION_DICT[0])
        else:
            ls.append(ACTION_DICT[1])
    action_ins = np.vstack(tuple(ls))
    ys = np.random.random(100)

    for i in range(100):
        Q.fit_single(np.random.random(4), np.random.randint(2), np.random.random())

    print(Q.pred_qval(np.random.random(4)))

