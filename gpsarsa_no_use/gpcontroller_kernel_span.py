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

    r_vec = np.array([0.0, 0.0])                # r:        reward vector
    # W = np.matrix(0.0)                        # W:        W matrix
    u_vec = np.array([0.0])                     # u:        mean vector
    C = np.matrix(0.0)                          # C:        covariance matrix

    def __init__(self, states, actions, initial_belief, initial_action):
        self.states = states
        self.actions = actions

        # array of tuples: [(b0, a0), (b1, a1), (b2, a2)]
        self.B = [(initial_belief, initial_action)]
        # self.K = np.array([[self.fullKernel_pair(initial_belief, initial_action)]])
        self.H = np.array([[1.0, -1.0*self.gamma]])
        self.dict = [(initial_belief, initial_action)]
        self.K_tilde_matrix = np.array([[self.fullKernel_pair(initial_belief, initial_action)]])
        k_tilde = self.getKVector(initial_belief, initial_action)
        self.G_matrix = np.array([[np.dot(np.linalg.inv(self.K_tilde_matrix), k_tilde)]])
        pass

    def get_best_action(self, belief):
        best = 0
        if np.random.sample() <= self.epsilon():
            # epsilon-greedy with 0.1 taking random action
            print '<<<epsilon random action...>>>'
            best = self.get_random_action()
        else:
            values = []
            for action in range(len(self.actions)):
                kvec = self.getKVector(belief, action)  # size: m
                values.append(np.dot(kvec, self.u_vec))
            pass
            print 'values: %s' % values
            v = np.amax(values)
            bests = []
            for action in range(len(self.actions)):
                if values[action] == v:
                    bests.append(action)
            best = np.random.choice(bests)
        return best

    def get_random_action(self):
        return np.random.choice(len(self.actions))

    def observe_step(self, old_belief, old_action, reward, new_belief, new_action, non_terminal=False):
        print 'old_belief: %s' % np.round(old_belief.flatten(), 3)
        if new_belief is not None:
            print 'new_belief: %s' % np.round(new_belief.flatten(), 3)
        print 'old_action: %s' % self.actions[old_action]
        if new_action is not None:
            print 'new_action: %s' % self.actions[new_action]
        print 'reward: %s' % reward

        B_prev = list(self.B)
        H_prev = np.copy(self.H)
        r_prev = np.copy(self.r_vec)
        K_tilde_prev = np.copy(self.K_tilde_matrix)
        G_prev = np.copy(self.G_matrix)

        # extend B matrix
        self.B.append((new_belief, new_action))

        if True:
            k_tilde = self.getKVector(new_belief, new_action)
            # delta 계산
            g_vec = np.dot(np.linalg.inv(self.K_tilde_matrix), k_tilde)
            delta = self.fullKernel_pair(new_belief, new_action) - np.dot(k_tilde, g_vec)

            if delta > self.nu:
                # expand dictionary
                self.dict.append((new_belief, new_action))
                # expand K gram matrix (m by m)
                self.K_tilde_matrix = self.extend_dim(self.K_tilde_matrix)
                k_tilde = self.getKVector(new_belief, new_action)

                len_m = len(self.dict)
                for ridx in range(len_m):
                    self.K_tilde_matrix[ridx][-1] = k_tilde[ridx]
                for cidx in range(len_m):
                    self.K_tilde_matrix[-1][cidx] = k_tilde[cidx]

            else:
                pass

            len_m = len(self.dict)
            len_t = len(self.B)
            # expand G gram matrix (t by m)
            self.G_matrix = np.zeros((len_t, len_m))
            for t in range(len_t):
                pair = self.B[t]
                temp_g = self.getKVector(pair[0], pair[1])
                for m in range(len_m):
                    self.G_matrix[t][m] = temp_g[m]

            # extend H matrix
            self.H = self.extend_dim(self.H)
            self.H[-1][-1] = -1 * self.gamma
            self.H[-1][-2] = 1
            pass
        # else:
        #     self.H = self.extend_row(self.H)
        #     self.H[-1][-1] = 1
        #     pass

        # append reward
        self.r_vec = np.r_[self.r_vec, reward]

        # update Q-function posterior
        H_tilde = np.dot(np.transpose(self.H), self.G_matrix)
        term1 = np.dot(np.dot(H_tilde, self.K_tilde_matrix), np.transpose(H_tilde))
        term2 = pow(self.sigma, 2) * np.dot(H_tilde, np.transpose(H_tilde))
        try:
            W_tilde = np.linalg.inv(term1+term2)
        except np.linalg.LinAlgError:
            print 'LingAlgError'
            self.B = list(B_prev)
            self.H = np.copy(H_prev)
            self.K_tilde_matrix = np.copy(K_tilde_prev)
            self.G_matrix = np.copy(G_prev)
            self.r_vec = np.copy(r_prev)
            return

        # mean vector
        term1 = np.dot(np.transpose(H_tilde), W_tilde)
        self.u_vec = np.dot(term1, self.r_vec)

        # covariance matrix
        self.C = np.dot(term1, H_tilde)

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
