# -*- coding: utf-8 -*-
import numpy as np
from math import *


class RandomController:
    def __init__(self, states, actions, initial_belief, initial_action):
        self.states = states
        self.actions = actions
        pass

    def get_best_action(self, belief):
        best = self.get_random_action()
        return best

    def get_random_action(self):
        return np.random.choice(len(self.actions))

    def observe_step(self, old_belief, old_action, reward, new_belief, new_action, non_terminal=False):
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
