import gym
import keras
import numpy as np
import random

from gym import wrappers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque

ACTIONS_DIM = 2
OBSERVATIONS_DIM = 4
MAX_ITERATIONS = 10 ** 6
LEARNING_RATE = 0.001

NUM_EPOCHS = 50

GAMMA = 0.99
REPLAY_MEMORY_SIZE = 1000
NUM_EPISODES = 10000
TARGET_UPDATE_FREQ = 100
MINIBATCH_SIZE = 32

RANDOM_ACTION_DECAY = 0.99
INITIAL_RANDOM_ACTION = 1


class ReplayBuffer():

    def __init__(self, max_size):
        self.max_size = max_size
        self.transitions = deque()

    def add(self, observation, action, reward, observation2):
        if len(self.transitions) > self.max_size:
            self.transitions.popleft()
        self.transitions.append((observation, action, reward, observation2))

    def sample(self, count):
        return random.sample(self.transitions, count)

    def size(self):
        return len(self.transitions)


def get_q(model, observation):
    np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])
    return model.predict(np_obs)


def train(model, observations, targets):
    np_obs = np.reshape(observations, [-1, OBSERVATIONS_DIM])
    np_targets = np.reshape(targets, [-1, ACTIONS_DIM])
    model.fit(np_obs, np_targets, epochs=1, verbose=0)


def predict(model, observation):
    np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])
    return model.predict(np_obs)


def get_model():
    model = Sequential()
    model.add(Dense(16, input_shape=(OBSERVATIONS_DIM,), activation='relu'))
    model.add(Dense(16, input_shape=(OBSERVATIONS_DIM,), activation='relu'))
    model.add(Dense(2, activation='linear'))

    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss='mse',
        metrics=[],
    )

    return model


def train_model(action_model, target_model, sample_transitions):
    random.shuffle(sample_transitions)
    batch_observations = []
    batch_targets = []

    for old_observation, action, reward, observation in sample_transitions:
        targets = np.reshape(get_q(action_model, old_observation), ACTIONS_DIM)
        targets[action] = reward
        if observation is not None:
            predictions = predict(target_model, observation)
            targets[action] += GAMMA * np.max(predictions)

        batch_observations.append(old_observation)
        batch_targets.append(targets)

    train(action_model, batch_observations, batch_targets)


def main():
    random_action_probability = INITIAL_RANDOM_ACTION
    # Initialize replay memory D to capacity N
    replay = ReplayBuffer(REPLAY_MEMORY_SIZE)
    action_model = get_model()
    env = gym.make('CartPole-v0')

    for episode in range(NUM_EPISODES):
        observation = env.reset()

        for iteration in range(MAX_ITERATIONS):
            random_action_probability *= RANDOM_ACTION_DECAY
            random_action_probability = max(random_action_probability, 0.1)
            old_observation = observation

            action = select_action(action_model, observation, random_action_probability)

            observation, reward, done, info = env.step(action)

            if done:
                print 'Episode {}, iterations: {}'.format(
                    episode,
                    iteration
                )

                reward = -200
                replay.add(old_observation, action, reward, None)
                break

            replay.add(old_observation, action, reward, observation)

            if replay.size() >= MINIBATCH_SIZE:
                sample_transitions = replay.sample(MINIBATCH_SIZE)
                train_model(action_model, action_model, sample_transitions)


def select_action(action_model, observation, random_action_probability):
    if np.random.random() < random_action_probability:
        return np.random.choice(range(ACTIONS_DIM))
    else:
        q_values = get_q(action_model, observation)
        return np.argmax(q_values)


if __name__ == "__main__":
    main()
