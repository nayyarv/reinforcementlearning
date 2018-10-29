"""
Modified version of Andrej Karparthy's pong from pixel to illustrate in a simple environment:
- the parameterisation of a policy
- the function approximation as a policy function
- applying the policy gradient update with the normalised reward
"""
import numpy as np
import gym

from polgradient.comtruise import MoveComtoBeacon

# hyperparameters
H = 1  # 200 # number of hidden layer neurons
batch_size = 1  # every how many episodes to do a param update?
learning_rate = 5e-2
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

rewardPlotArray = []

# model initialization
model = {}
model['W1'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


#
#  _____            _
# |_   _|___     __| | ___
#   | | / _ \   / _` |/ _ \
#   | || (_) | | (_| | (_) |
#   |_| \___/   \__,_|\___/
#
# - Complete the function that'll discount a list of rewards and return the discounted rewards
#
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    return discounted_r


def policy_forward(x):
    h = x
    logp = np.dot(model['W1'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW1 = np.dot(eph.T, epdlogp).ravel()
    return {'W1': dW1}


class Agent:
    def __init__(self):
        self.env = MoveComtoBeacon()

    def train(self):
        state = self.env.reset()
        episode_number = 0

        prev_x = None  # used in computing the difference frame
        xs, hs, dlogps, drs = [], [], [], []
        running_reward = 0
        reward_sum = 0
        episode_number = 0
        episodeLengthCounter = 0
        running_rewardArray = []

        while episode_number < 100:
            # forward the policy network and sample an action from the returned probability
            aprob, h = policy_forward(state)

            action = 0 if np.random.uniform() < aprob else 1  # randomly take 1 of two actions. we are sampling from a bernoulli distribution here

            # record various intermediates (needed later for backprop)
            xs.append(state)  # observation
            hs.append(h)  # hidden state

            y = 1 if action == 0 else 0  # a "fake label"

            # we want to head in the opposite direction to the action we took as that is the gradient
            dlogps.append(
                y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

            # step the environment and get new measurements
            state, reward, done, info = self.env.step(action)
            reward_sum += reward

            drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

            episodeLengthCounter += 1

            if done:  # an episode finished

                rewardPlotArray.append(reward_sum)

                episodeLengthCounter = 0

                episode_number += 1

                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                epx = np.vstack(xs)  # vstack changes a row of data into a column.
                eph = np.vstack(hs)
                epdlogp = np.vstack(dlogps)

                epr = np.vstack(drs)
                xs, hs, dlogps, drs = [], [], [], []  # reset array memory

                # call the discount_rewards function to compute the discounted reward backwards through time
                discounted_epr = discount_rewards(epr)
                #
                #  _____            _
                # |_   _|___     __| | ___
                #   | | / _ \   / _` |/ _ \
                #   | || (_) | | (_| | (_) |
                #   |_| \___/   \__,_|\___/
                #
                # - normalise the discounted_epr variable which is the discounted reward

                epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
                grad = policy_backward(eph, epdlogp)

                for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

                # update our neural network once we have reached the batch size &
                # perform rmsprop parameter update every batch_size episodes
                if episode_number % batch_size == 0:
                    for k, v in model.items():
                        g = grad_buffer[k]  # gradient
                        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                        grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

                # boring book-keeping
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                running_rewardArray.append(running_reward)

                reward_sum = 0
                state = self.env.reset()  # reset env

            return running_rewardArray

    def plot(self, running_rewardArray):
        x = np.linspace(0, len(running_rewardArray), len(running_rewardArray))

        actionMemPlot = plt.figure()

        plt.xlabel("time")
        plt.ylabel("reward")
        plt.plot(x, running_rewardArray, linewidth=2)

    def run(self):
        """
        Run the agent to see it work
        """
        from gym.wrappers import Monitor
        env = Monitor(self.env, './video', force=True)
        state = env.reset()
        reward_sum = 0
        episode_number = 0
        while episode_number < 2:
            # forward the policy network and sample an action from the returned probability
            aprob, h = policy_forward(state)

            action = 0 if np.random.uniform() < aprob else 1  # randomly take 1 of two actions. we are sampling from a bernoulli distribution here

            # step the environment and get new measurements
            state, reward, done, info = env.step(action)
            reward_sum += reward
            env.render()

            if done:  # an episode finished
                episode_number += 1
                print("Episode finished with total reward", reward_sum)
                reward_sum = 0
                state = env.reset()  # reset env
