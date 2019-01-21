import bisect
import random

import gym
import numpy as np

env = gym.make('CartPole-v0')


def collect_data(n_trials):
    data = []
    for i_episode in range(n_trials):
        observation = env.reset()
        for t in range(100):
            action = env.action_space.sample()
            prev_observation = observation
            observation, reward, done, info = env.step(action)
            data.append((prev_observation, observation, action, done))
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    return data


def calculate_bounds_old(feature_data, count):
    l = sorted(feature_data)
    jump = len(l) / count
    return [l[i] for i in range(jump, len(l), jump)][:count - 1]


def calculate_bounds(feature_data, count):
    m1 = np.min(feature_data)
    m2 = np.max(feature_data)
    jump = (m2 - m1) / count
    return list(np.arange(m1 + jump, m2, jump))[:count - 1]


class Model(object):
    def __init__(self, count, data):
        n = pow(count, len(data[0][0]))
        self._n_cells = n
        self._transfer_probs0 = np.zeros((n, n), dtype=np.float)
        self._transfer_probs1 = np.zeros((n, n), dtype=np.float)
        self._survival_probs = np.zeros((n,), dtype=np.float)
        self._smart_survival_probs = []
        self._construct(data, count)

    def _construct(self, data, count):
        self._cells = self._calculate_cells(np.array([o for _, o, _, _ in data]), count)
        transfers = self._reduce_to_cells(data)
        # print([t[0] for t in transfers])
        self._construct_transfer_probs([(o1, o2, action) for o1, o2, action, done in transfers])
        self._construct_survival_probs([(o2, done) for o1, o2, action, done in transfers])

    def _calculate_cells(self, arr, count):
        cells = []
        for feature_data in arr.T:
            bounds = calculate_bounds(feature_data, count)
            # print("bounds", bounds)
            assert len(bounds) == count - 1, (len(bounds), count, min(feature_data), max(feature_data))
            cells.append(bounds)
        return cells

    def _reduce_to_cells(self, data):
        return [(self._reduce_observation(o1), self._reduce_observation(o2), action, done)
                for o1, o2, action, done in data]

    def _reduce_observation(self, o):
        assert len(o) == len(self._cells)
        indices = [bisect.bisect_left(bounds, number) for number, bounds in zip(o, self._cells)]
        count = len(self._cells[0]) + 1
        return sum(x * pow(count, i) for i, x in enumerate(indices))

    def _construct_survival_probs(self, transfers):
        probs = self._survival_probs = self._first_order_survival_probs(transfers)
        for _ in range(200):
            probs = self._induce_survival_probs(probs)
            self._smart_survival_probs.append(probs)

    def _first_order_survival_probs(self, transfers):
        deaths_count = [global_deaths_init] * self._n_cells
        survival_count = [global_deaths_init] * self._n_cells
        for cell, done in transfers:
            if done:
                deaths_count[cell] += 1
            else:
                survival_count[cell] += 1
        return np.array(
            [float(survival) / (survival + death) for survival, death in zip(survival_count, deaths_count)])

    def _construct_transfer_probs(self, transfers):
        self._transfer_probs0 += global_transfer_probs_init
        self._transfer_probs1 += global_transfer_probs_init
        for o1, o2, action in transfers:
            if action == 0:
                self._transfer_probs0[o1, o2] += 1
            else:
                self._transfer_probs1[o1, o2] += 1
        self._transfer_probs0 = self._normalize_rows(self._transfer_probs0)
        self._transfer_probs1 = self._normalize_rows(self._transfer_probs1)

    def _normalize_rows(self, arr):
        row_sums = arr.sum(axis=1)
        return arr / row_sums[:, np.newaxis]

    def _induce_survival_probs(self, probs):
        step0_survival = np.dot(self._transfer_probs0, probs)
        step1_survival = np.dot(self._transfer_probs1, probs)
        if np.average(step0_survival) > np.average(step1_survival):
            return step0_survival
        else:
            return step1_survival

    def select_step(self, o, depth):
        cell = self._reduce_observation(o)
        A0 = np.dot(self._transfer_probs0[cell],
                    self._smart_survival_probs[depth])
        A1 = np.dot(self._transfer_probs1[cell],
                    self._smart_survival_probs[depth])
        return int(np.average(A1) > np.average(A0))


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


# data = [create() for _ in range(10000)]
# # print([o[0] for o in data])
# m = Model(3, data)
# print(m._transfer_probs0)
# print(m._transfer_probs1)
# print(m._smart_survival_probs[90])
# for i in range(10):
#     print(m.select_step([i / 5. + 0.5, i / 3.], 90))

def play(data, n_trials):
    global global_transfer_probs_init
    global global_deaths_init
    scores = []

    for i_episode in range(n_trials):
        observation = env.reset()
        global_transfer_probs_init = 0.01
        global_deaths_init = 0.01
        model = Model(4, data)
        for t in range(200):
            action = model.select_step(observation, 180)
            prev_observation = observation
            observation, reward, done, info = env.step(action)
            data.append((prev_observation, observation, action, done))
            if done:
                break
        scores.append(t)
        print(i_episode, np.average(scores[-100:]))
    return scores


for t_init in (0.25, 0.125, 0.05, 0.01):
    for d_init in (0.25, 0.125, 0.05, 0.01):
        data = collect_data(1000)
        print("t_init", t_init)
        print("d_init", d_init)
        global_transfer_probs_init = t_init
        global_deaths_init = d_init
        scores = play(data, 2000)[-100:]
        print(np.average(scores))
