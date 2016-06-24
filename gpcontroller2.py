# -*- coding: utf-8 -*-
import numpy as np
from math import *


class GPController:
    # Meta parameters
    nu = 0.01
    sigma = 14.12
    gamma = 0.9

    # Gaussian Kernel parameters
    # sigma_state = 0.2
    # c_state = 10
    # b_action = 0.1
    kernel_sigma = 2
    kernel_p = 2

    # Linear Kernel parameters
    # lkernel_p = 2
    # lkernel_sigma = 1

    K_tilde_inv = None              # initialized in __init__ method
    a = np.array([1.0])             # a:            alpha vector
    alpha_tilde = np.array([0.0])   # alpha_tilde:  alpha tilde vector
    C_tilde = np.matrix(0.0)        # C_tilde:      c tilde matrix
    c_tilde = np.zeros(1)           # c_tilde:      c tilde vector
    d = 0.0
    s = float('inf')

    def __init__(self, states, actions, initial_belief, initial_action):
        self.states = states
        self.actions = actions

        # array of tuples: [(b0, a0), (b1, a1), (b2, a2)]
        self.dict = [(initial_belief, initial_action)]
        self.K_tilde_inv = np.array([[1.0 / self.fullKernel_pair(initial_belief, initial_action)]])

    def get_best_action(self, belief):
        best = 0
        if np.random.sample() <= self.epsilon():
            best = self.get_random_action()
        else:
            values = []
            for action in range(len(self.actions)):
                k_tilde = self.getKVector(belief, action)
                v = np.dot(k_tilde, self.alpha_tilde)
                values.append(v)
            v = np.amax(values)
            bests = []
            for action in range(len(self.actions)):
                if values[action] == v:
                    bests.append(action)
            best = np.random.choice(bests)

            print 'values: %s' % values
            # print 'best action: %s' % self.actions[best]
        return best

    def get_random_action(self):
        return np.random.choice(len(self.actions))

    def observe_step(self, old_belief, old_action, reward, new_belief, new_action):
        print 'new_belief: %s' % np.round(new_belief.flatten(), 3)
        print 'old_action: %s' % self.actions[old_action]
        print 'new_action: %s' % self.actions[new_action]
        print 'reward: %s' % reward
        return

        k_tilde = self.getKVector(new_belief, new_action)
        a_prev = self.a
        self.a = np.dot(self.K_tilde_inv, k_tilde)
        delta = self.fullKernel_pair(new_belief, new_action) - np.dot(k_tilde, self.a)

        k_tilde_prev = self.getKVector(old_belief, old_action)
        delta_k_tilde = k_tilde_prev - k_tilde * self.gamma
        _lambda = self.gamma * pow(self.sigma, 2) / self.s
        self.d = self.d * _lambda + reward - np.dot(delta_k_tilde, self.alpha_tilde)

        print 'delta: %d' % delta
        if delta > self.nu:
            """
            delta가 threshold nu를 초과할 경우, dictionary 확장
            """
            # TODO: replace 2d-array augmentation with PANDAS framework
            K_tilde_inv_prev = self.K_tilde_inv
            self.K_tilde_inv = self.K_tilde_inv * delta + np.dot(self.a, self.a)
            self.K_tilde_inv = self.extend_dim(self.K_tilde_inv)        # extend dim
            len_t = len(K_tilde_inv_prev)
            # assign values to new column
            for ridx in range(len_t):
                self.K_tilde_inv[ridx][-1] = -1 * self.a[ridx]
            # assign values to new row
            for cidx in range(len_t):
                self.K_tilde_inv[-1][cidx] = -1 * self.a[cidx]
            # last element
            self.K_tilde_inv[-1][-1] = 1.0
            self.K_tilde_inv = self.K_tilde_inv / delta

            self.a = np.zeros(len(self.a)+1)
            self.a[-1] = 1
            h_tilde = np.r_[a_prev, (-1*self.gamma)]
            delta_k = np.dot(a_prev, (k_tilde_prev - k_tilde * 2.0 * self.gamma))
            delta_k += pow(self.gamma, 2) * self.fullKernel_pair(new_belief, new_action)

            # calc for new s
            self.s = (1+pow(self.gamma, 2)) * pow(self.sigma, 2) + delta_k \
                - np.dot(np.dot(delta_k_tilde, self.C_tilde), delta_k_tilde).getA1() \
                + 2 * _lambda * np.dot(self.c_tilde, delta_k_tilde) \
                - _lambda * self.gamma * pow(self.sigma, 2)
            self.s = self.s.item(0)

            # calc for new c_tilde
            temp1 = np.zeros(len(self.c_tilde)+1)
            temp2 = np.zeros(len(self.c_tilde)+1)
            temp1[:-1] = self.c_tilde   # it's a deep copy of c_tilde array
            temp1[:-1] = np.dot(self.C_tilde, delta_k_tilde).getA1()

            self.c_tilde = np.r_[self.c_tilde, 0]
            self.c_tilde[:] = temp1 * _lambda + h_tilde - temp2

            # calc for alpha_tilde
            self.alpha_tilde = np.r_[self.alpha_tilde, 0]   # simple augmenting

            # calc for C_tilde
            self.C_tilde = self.extend_dim(self.C_tilde)
            self.dict.append((new_belief, new_action))

        else:
            """
            delta가 threshold nu를 초과하지 않을 경우, dictionary를 확장하지 않음
            """
            h_tilde = a_prev - (self.a * self.gamma)
            delta_k = np.dot(h_tilde, delta_k_tilde)
            c_tilde_prev = self.c_tilde
            self.c_tilde = self.c_tilde * _lambda + h_tilde - np.dot(self.C_tilde, delta_k_tilde).getA1()
            if isinf(self.c_tilde[0]):
                print 'c_tilde positive infinity break'

            # calc for new s
            self.s = (1+pow(self.gamma, 2)) * pow(self.sigma, 2) \
                + np.dot(delta_k_tilde, (self.c_tilde + c_tilde_prev * _lambda)) \
                - _lambda * self.gamma * pow(self.sigma, 2)
            self.s = self.s.item(0)
        # end of if

        self.alpha_tilde = self.alpha_tilde + (self.c_tilde / self.s) * self.d
        self.C_tilde = self.C_tilde + np.dot(self.c_tilde, self.c_tilde) / self.s

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
        v = -1 * (np.linalg.norm(b1-b2)) / (2 * pow(self.kernel_sigma, 2))
        result = pow(self.kernel_p, 2) * exp(v)

        # Linear kernel with param (sigma 2, p=3)
        # v = np.dot(np.transpose(b1), b2) + pow(self.lkernel_sigma, 2)
        # result = pow(v, self.lkernel_p)
        return result

    def actionKernel_pair(self, action):
        return self.actionKernel(action, action)

    def actionKernel(self, a1, a2):
        """
        Kernel of two action a1, a2, return value will be a scala
        """
        # for simplicity just using delta-kernel
        return 0.0001 if a1 == a2 else 1.0

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

    @staticmethod
    def epsilon():
        return 1
