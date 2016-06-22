import numpy as np
import math


class GPController:
    # Meta parameters
    nu = 0.1
    sigma = 1
    gamma = 0.9

    # Kernel parameters
    # sigma_state = 0.2
    # c_state = 10
    # b_action = 0.1
    kernel_sigma = 5
    kernel_p = 4

    dict = []                       # array of tuples: [(b0, a0), (b1, a1), (b2, a2)]
    K_tilde_inv = None              # initialized in __init__ method
    a = np.array([1.0])             # a:            alpha vector
    alpha_tilde = np.array([0.0])   # alpha_tilde:  alpha tilde vector
    C_tilde = np.matrix(0.0)        # C_tilde:      c tilde matrix
    c_tilde = np.zeros(1)           # c_tilde:      c tilde vector
    d = 0.0
    s = float('inf')

    def __init__(self, states, actions, initial_state, initial_action):
        self.states = states
        self.actions = actions

        self.K_tilde_inv = np.array([1.0 / self.__fullKernel(initial_state, initial_action)])
        pass

    def get_best_action(self):
        return np.random.choice(len(self.actions))
        pass

    def observe_step(self, old_belief, old_action, reward, new_belief, new_action):
        print 'new_belief: %s' % np.round(new_belief.flatten(), 3)
        print 'new_action: %d' % new_action
        print 'reward: %s' % reward

        k_tilde = self.getKVector(new_belief, new_action)
        a_prev = self.a
        self.a = np.dot(self.K_tilde_inv, k_tilde)
        delta = self.__fullKernel(new_belief, new_action) - np.dot(k_tilde, self.a)
        k_tilde_prev = self.getKVector(old_belief, old_action)
        delta_k_tilde = k_tilde_prev - k_tilde * self.gamma
        _lambda = self.gamma * math.pow(self.sigma, 2) / self.s

        if delta > self.nu:
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
            delta_k += math.pow(self.gamma, 2) * self.__fullKernel(new_belief, new_action)

            # calc for new s
            self.s = (1+math.pow(self.gamma, 2)) * self.sigma + delta_k \
                - np.dot(np.dot(delta_k_tilde, self.C_tilde), delta_k_tilde) \
                + 2 * _lambda * np.dot(self.c_tilde, delta_k_tilde) \
                - _lambda * self.gamma * math.pow(self.sigma, 2)

            # calc for new c_tilde
            temp1 = np.zeros(len(self.c_tilde)+1)
            temp2 = np.zeros(len(self.c_tilde)+1)
            temp1[:-1] = self.c_tilde   # it's a deep copy of c_tilde array
            temp1[:-1] = np.dot(self.C_tilde, delta_k_tilde)
            self.c_tilde[:] = temp1 * _lambda + h_tilde - temp2

            # calc for alpha_tilde
            self.alpha_tilde = np.r_[self.alpha_tilde, 0]   # simple augmenting

            # calc for C_tilde
            self.C_tilde = self.extend_dim(self.C_tilde)

        else:
            # TODO: CONTINUE HERE TOMORROW...
            pass

    def getKVector(self, new_belief, new_action):
        k = np.zeros(len(self.dict))
        for i in range(len(k)):
            k[i] = self.fullKernel(self.dict[i][0], new_belief,
                                   self.dict[i][1], new_action)
        return k

    def __fullKernel(self, belief, action):
        # return self.__stateKernel(belief) * self.__actionKernel(action)
        return 1    # dummy just for now

    def fullKernel(self, b1, b2, a1, a2):
        rv = self.stateKernel(b1, b2) * self.actionKernel(a1, a2)
        return 1    # dummy just for now

    def __stateKernel(self, belief):
        return self.stateKernel(belief, belief)

    def stateKernel(self, b1, b2):
        """
        Kernel of two belief vector b1, b2, return value will be a scala
        """
        # Gaussian kernel with param (sigma=5, p=4)
        v = -1 * (np.linalg.norm(b1-b2)) / (2 * math.pow(self.kernel_sigma, 2))
        result = math.pow(self.kernel_p, 2) * math.exp(v)
        print 'state kernel result: %d' % result
        pass

    def __actionKernel(self, action):
        return self.actionKernel(action, action)

    def actionKernel(self, a1, a2):
        """
        Kernel of two action a1, a2, return value will be a scala
        """
        # for simplicity just using delta-kernel
        return 0.0 if a1 == a2 else 1.0

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
        return 0.1
