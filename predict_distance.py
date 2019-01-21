import random

import gym
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential

env = gym.make('CartPole-v0')


def collect_data(n_trials):
    memory = Memory(None)
    for i_episode in range(n_trials):
        observation = env.reset()
        memory.set_current_observation(observation)
        for t in range(100):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            memory.remember(observation, action, done)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    return memory


class Memory(object):
    def __init__(self, observation):
        self._data = []
        self._last_observation = observation

    def set_current_observation(self, observation):
        self._last_observation = observation

    def remember(self, observation, action, done):
        self._data.append((self._last_observation, observation, action, done))
        self._last_observation = observation

    def steps_data(self):
        return [(o1, o2, action) for o1, o2, action, done in self._data]

    def is_done_data(self):
        return [(o2, done) for o1, o2, action, done in self._data]

    def score_data(self):
        res = []
        done_indices = [i for i, (_, _, _, done) in enumerate(self._data) if done]
        for i_start, i_end in zip(done_indices[:-1], done_indices[1:]):
            for i in range(max(i_end - 30, i_start + 1), i_end):
                distance = i_end - i
                res.append((self._data[i][1], distance))
        return res


class Model(object):
    def __init__(self):
        self._step_model = self._create_step_model()
        self._score_model = self._create_score_model()

    def train(self, memory, epochs):
        self.train_steps(memory, epochs)
        self.train_score(memory, epochs)

    def train_score(self, memory, epochs):
        x = np.array([o for o, score in memory.score_data()])
        y = np.array([score for o, score in memory.score_data()], dtype=np.float)
        self._score_model.fit(x, y, epochs=epochs, verbose=0)
        print("distance loss", self._score_model.evaluate(x, y, verbose=0))

    def train_steps(self, memory, epochs):
        x = np.array([self._extend_observation(o1, action) for o1, o2, action in memory.steps_data()])
        print(x.shape)
        y = np.array([o2 for o1, o2, action in memory.steps_data()])
        perm = np.random.permutation(len(x))
        x = x[perm]
        y = y[perm]
        self._step_model.fit(x, y, epochs=epochs, verbose=0)
        print("steps loss", self._step_model.evaluate(x, y, verbose=0))

    def _extend_observation(self, o, action):
        return np.concatenate((o, np.array([action], dtype=np.float32)))

    def select_step(self, o):
        scores = []
        for action in (0, 1):
            o_next = self._step_model.predict(np.reshape(self._extend_observation(o, action), (1, 5)))[0]
            score = self._score_model.predict(np.reshape(o_next, [1, 4]))[0][0]
            scores.append(score)
        return np.argmax(scores)

    def _create_score_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim=4, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(48, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(48, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def _create_step_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim=5, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(48, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(48, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model


def create():
    x = random.random() * 3
    y = random.random() * 3
    action = random.randint(0, 1)
    old = np.array([x, y])
    if action == 0:
        new = old / 2.
    else:
        new = old
    return old, new, action, new[0] < 0.5


def play(memory, n_trials):
    scores = []
    model = Model()
    model.train(memory, 100)
    for i_episode in range(n_trials):
        observation = env.reset()
        memory.set_current_observation(observation)
        for t in range(200):
            action = model.select_step(observation)
            observation, reward, done, info = env.step(action)
            memory.remember(observation, action, done)
            if done:
                break
        scores.append(t)
        print(scores[-20:])
        print("episode", i_episode, "avg score", np.average(scores[-100:]))


for N in (5, 10, 25):
    print("N=", N)
    memory = collect_data(N)
    play(memory, 100)
