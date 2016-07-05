# -*- coding: utf-8 -*-
"""
copyright Kwangho Heo
2016-07-04
"""
import numpy as np
from math import *
import util


class GPTDController:

    steps = 0

    # Meta parameters
    nu = 0.1
    sigma0 = 10.0
    gamma = 0.9

    # Gaussian Kernel parameters
    kernel_sigma = 0.5
    kernel_p = 2.0

    # Linear Kernel parameters
    lkernel_p = 1
    lkernel_sigma = 0

    tmp_action_values = []  # just for print

    def __init__(self, states, actions, initial_belief, initial_action, fixed_epsilon=True):

        self.fixed_epsilon = fixed_epsilon

        self.states = states
        self.actions = actions

        self.C = util.GrowingMat((0, 1), (100, 100))
        self.c = util.GrowingVector(0)
        self.alpha = util.GrowingVector(0)
        self.d = 0
        self.sinv = 0
        self.Kinv = util.GrowingMat((0, 1), (100, 100))

        # array of tuples: [(b0, a0), (b1, a1), (b2, a2)]
        self.dict = [(initial_belief, initial_action)]
        self.Kinv.expand(rows=np.array([[1. / self.fullKernel_pair(initial_belief, initial_action)]]))
        self.C.expand(rows=np.array(0))
        self.a = np.array(1)
        self.c.expand(rows=np.array(0))
        self.alpha.expand(rows=np.array(0))
        pass

    def get_best_action(self, belief):
        if np.random.sample() <= self.epsilon(self.steps):
            # epsilon-greedy with 0.1 taking random action
            best = self.get_random_action()
            # print '<<<epsilon random action...>>>'
        else:
            values = []
            for action in range(len(self.actions)):
                kvec = self.getKVector(belief, action)      # size: m
                values.append(float(np.inner(kvec.T, self.alpha.view.flatten())))
            pass
            print 'values: %s' % [float(np.round(v, 4)) for v in values]
            v = np.amax(values)
            bests = []
            for action in range(len(self.actions)):
                if values[action] == v:
                    bests.append(action)
            best = np.random.choice(bests)
            self.tmp_action_values = values
        # best = self.get_random_action()
        return best

    def get_random_action(self):
        return np.random.choice(len(self.actions))

    def observe_step(self, old_belief, old_action, reward, new_belief, new_action, non_terminal=False):
        self.steps += 1
        print 'old_belief: %s' % np.round(old_belief.flatten(), 3)
        print 'old_action: %s' % self.actions[old_action]
        print 'new_belief: %s' % np.round(new_belief.flatten(), 3)
        print 'new_action: %s' % self.actions[new_action]
        print 'reward: %s' % reward
        # print 'new action values %s' % [float(np.round(v, 4)) for v in self.tmp_action_values]

        k = self.getKVector(new_belief, new_action)
        a = np.array(np.dot(self.Kinv.view, k)).flatten()
        ktt = float(self.fullKernel_pair(new_belief, new_action))
        dk = self.getKVector(old_belief, old_action) - self.gamma * self.getKVector(new_belief, new_action)
        delta = ktt - float(np.inner(k.T, a))
        self.d = self.d * self.sinv * self.gamma * self.sigma0 ** 2 + \
            reward - float(np.inner(dk, self.alpha.view.flatten()))

        print 'delta: %f' % delta
        # sparsification test
        if delta > self.nu:
            dk2 = np.array((self.getKVector(old_belief, old_action) - 2 *
                            self.gamma * self.getKVector(new_belief, new_action))).flatten()
            self.dict.append((new_belief, new_action))
            # update K^-1
            self.Kinv.view = delta * self.Kinv.view + np.outer(a, a)
            self.Kinv.expand(cols=-a.reshape(-1, 1),
                             rows=-a.reshape(1, -1),
                             block=np.array([[1]])
                             )
            self.Kinv.view /= delta
            print "inverted Kernel matrix:", self.Kinv.view

            a = np.zeros(self.Kinv.shape[0])
            a[-1] = 1

            hbar = np.zeros_like(a)
            hbar[:-1] = self.a
            hbar[-1] = - self.gamma

            dktt = float(np.inner(self.a, dk2)) + self.gamma ** 2 * ktt

            cm1 = self.c.view.copy().flatten()
            self.c.view = self.c.view.flatten() * self.sinv * self.gamma * self.sigma0 ** 2 \
                + self.a - np.dot(self.C.view, dk)
            self.c.expand(rows=np.array(- self.gamma))

            s = (1 + self.gamma ** 2) * self.sigma0 ** 2 - self.sinv * self.gamma ** 2 * self.sigma0 ** 4 + dktt \
                - np.dot(dk, np.dot(self.C.view, dk)) + 2 * self.sinv * self.gamma * self.sigma0 ** 2 * np.dot(cm1, dk)

            self.alpha.expand(rows=np.array([[0]]))
            self.C.expand(rows=np.zeros((1, self.C.shape[1])),
                          cols=np.zeros((self.C.shape[0], 1)))
            pass

        else:
            self.hbar = self.a - self.gamma * a
            cm1 = self.c.view.copy()
            self.c.view = self.c.view.flatten() * self.sinv * self.gamma * self.sigma0 ** 2 + self.hbar \
                - np.dot(self.C.view, dk)

            s = (1 + self.gamma ** 2) * self.sigma0 ** 2 - self.sinv * self.gamma ** 2 * self.sigma0 ** 4 + \
                np.dot(dk, self.c.view + self.gamma * self.sigma0 ** 2 * self.sinv * cm1)
            pass

        self.sinv = 1 / s
        self.alpha.view += self.sinv * self.d * self.c.view
        self.C.view += self.sinv * np.outer(self.c.view, self.c.view)
        self.a = a
        pass

    def getKVector(self, new_belief, new_action):
        k = np.zeros(len(self.dict))
        for i in range(len(k)):
            k[i] = self.fullKernel(self.dict[i][0], new_belief,
                                   self.dict[i][1], new_action)
        return k

    def fullKernel_pair(self, belief, action):
        return self.stateKernel_pair(belief) + self.actionKernel_pair(action)

    def fullKernel(self, b1, b2, a1, a2):
        return self.stateKernel(b1, b2) + self.actionKernel(a1, a2)

    def stateKernel_pair(self, belief):
        return self.stateKernel(belief, belief)

    def stateKernel(self, b1, b2):
        """
        Kernel of two belief vector b1, b2, return value will be a scala
        """
        # Gaussian kernel with param (sigma=5, p=4)
        # v = - (np.linalg.norm(b1-b2) ** 2) / (2 * self.kernel_sigma ** 2)
        # result = pow(self.kernel_p, 2) * exp(v)

        # scaled norm kernel
        result = 1 - np.linalg.norm(b1-b2) ** 2 / (np.linalg.norm(b1) ** 2 * np.linalg.norm(b2) ** 2)
        return result

    def actionKernel_pair(self, action):
        return self.actionKernel(action, action)

    def actionKernel(self, a1, a2):
        """
        Kernel of two action a1, a2, return value will be a scala
        """
        # for simplicity just using delta-kernel
        result = 1 if a1 == a2 else 0
        return result

    def extend_dim(self, dim2arr):
        """
        extends rows and columns by 1
        :param dim2arr: 2D-Array
        :return:
        """
        col = np.zeros(dim2arr.shape[0])
        dim2arr = np.column_stack((dim2arr, col))
        row = np.zeros(dim2arr.shape[1])
        dim2arr = np.row_stack((dim2arr, row))
        return dim2arr

    def end(self):
        print 'end debug here'
        print 'dictionary length: %d' % len(self.dict)

    def epsilon(self, steps):
        if self.fixed_epsilon:
            return 0.1
        else:
            e = 0.2 / np.log10(steps+10)
            print 'epsilon: %f' % e
            return e
