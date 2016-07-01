# -*- coding: utf-8 -*-
import numpy as np
from math import *


class GPController:
    # Meta parameters
    nu = 0.1
    sigma = 2
    gamma = 0.9

    # Gaussian Kernel parameters
    kernel_sigma = 2.0
    kernel_p = 1.0

    # Linear Kernel parameters
    lkernel_p = 1
    lkernel_sigma = 0

    r_vec = np.array([0.0])             # r:        reward vector
    # W = np.matrix(0.0)                # W:        W matrix
    u_vec = np.array([0.0])             # u:        mean vector
    C = np.matrix(0.0)                  # C:        covariance matrix

    def __init__(self, states, actions, initial_belief, initial_action):
        self.states = states
        self.actions = actions

        # array of tuples: [(b0, a0), (b1, a1), (b2, a2)]
        self.B = [(initial_belief, self.actions[initial_action])]
        self.K = np.array([[self.fullKernel_pair(initial_belief, initial_action)]])
        self.H = np.array([[1.0, -1.0*self.gamma]])

        pass

    def get_best_action(self, belief):
        best = 0
        if np.random.sample() <= self.epsilon():
            # epsilon-greedy with 0.1 taking random action
            print '<<<epsilon random action...>>>'
            best = self.get_random_action()
        else:
            # values = []
            # for action in range(len(self.actions)):
            #     kvec = self.getKVector(belief, action)  # size: m
            #     values.append(np.dot(kvec, self.u_vec))
            # pass
            # print 'values: %s' % values
            # v = np.amax(values)
            # bests = []
            # for action in range(len(self.actions)):
            #     if values[action] == v:
            #         bests.append(action)
            # best = np.random.choice(bests)
            best = self.get_random_action()
        return best

    def get_random_action(self):
        return np.random.choice(len(self.actions))

    def observe_step(self, old_belief, old_action, reward, new_belief, new_action, non_terminal=False):
        print 'old_belief: %s' % np.round(old_belief.flatten(), 3)
        print 'new_belief: %s' % np.round(new_belief.flatten(), 3)
        print 'old_action: %s' % self.actions[old_action]
        print 'new_action: %s' % self.actions[new_action]
        print 'reward: %s' % reward
        self.B.append((new_belief, self.actions[new_action]))
        self.r_vec = np.r_[self.r_vec, reward]


    def getKVector(self, new_belief, new_action):
        k = np.zeros(len(self.B))
        for i in range(len(k)):
            k[i] = self.fullKernel(self.B[i][0], new_belief,
                                   self.B[i][1], new_action)
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
        # v = -1.0 * (np.linalg.norm(b1-b2)) / (2 * pow(self.kernel_sigma, 2))
        # result = pow(self.kernel_p, 2) * exp(v)

        # Linear kernel with param (sigma 2, p=3)
        # v = np.dot(np.transpose(b1), b2) + pow(self.lkernel_sigma, 2)
        # result = pow(v, self.lkernel_p)
        result = np.dot(np.transpose(b1), b2).item(0)
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

    def extend_row(self, dim2arr):
        row = np.zeros(dim2arr.shape[1])
        dim2arr = np.row_stack((dim2arr, row))
        return dim2arr

    def end(self):
        print 'end debug here'
        pass

    @staticmethod
    def epsilon():
        return 0.1
