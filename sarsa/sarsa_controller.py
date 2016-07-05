# -*- coding: utf-8 -*-
import numpy as np
import random
import collections


class SarsaController:

    # belief vector quantize size e.g. (0~0.05) -> 0,      (0.05001 ~ 0.1)
    # 20 x 20
    grid_size = 0.05
    gamma = 0.9
    # learning rate
    alpha = 0.5

    # learning step count
    steps = 0

    tmp_action_values = []      # just for print

    def __init__(self, states, actions, initial_belief, initial_action, fixed_epsilon=True):
        self.fixed_epsilon = fixed_epsilon

        self.states = states
        self.actions = actions

        self.B = [(initial_belief, initial_action)]
        self.Q = collections.OrderedDict()
        print 'initial action: %s' % self.actions[initial_action]
        pass

    def get_best_action(self, belief):
        if np.random.random() <= self.epsilon(self.steps):
            print '>>> epsilon greedy action'
            best = self.get_random_action()
        else:
            belief_tup = self.quantize_belief(belief)
            best = self.getMaxQ(belief_tup)[1]
            print 'maxQ action: %s' % self.actions[best]
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
        print 'new action values %s' % [float(np.round(v, 2)) for v in self.tmp_action_values]

        # Q(S, A) + alpha[R + gamma*maxQ(S', a) - Q(S, A)]
        old_belief_tup = self.quantize_belief(old_belief)
        new_belief_tup = self.quantize_belief(new_belief)

        q_prime = self.getQ(new_belief_tup, new_action)
        q_value = self.getQ(old_belief_tup, old_action)
        new_q = q_value + self.alpha * (reward + self.gamma * q_prime) - q_value
        self.Q[(old_belief_tup, old_action)] = new_q
        pass

    def getMaxQ(self, belief_tup):
        max = -10000
        values = []
        for action in range(len(self.actions)):
            values.append(self.getQ(belief_tup, action))
        max = np.amax(values)
        bests = []
        for action in range(len(self.actions)):
            if values[action] == max:
                bests.append(action)
        best = np.random.choice(bests)
        self.tmp_action_values = values
        return max, best

    def getQ(self, belief_tup, action):
        # random.uniform(-2, 2)
        # print 'getQ: %s, %s' % (belief_tup[0], belief_tup[1])
        try:
            return self.Q[(belief_tup, action)]
        except KeyError:
            # return 0.0
            return random.uniform(-2.0, 2.0)

    def quantize_belief(self, belief):
        sec1 = self.quantize_value(belief[0].item(0))
        sec2 = self.quantize_value(belief[1].item(0))
        return sec1, sec2
        pass

    def quantize_value(self, value):
        """
        returns vector grid tuple
        """
        section = 0
        for i in range(int(1/0.05)+1):
            if self.grid_size * i >= value:
                section = i
                break
        lower = np.round(self.grid_size*(section-1), 3)
        upper = np.round(self.grid_size*section, 3)
        # print '%.2f\'s grid is: [%f~%f)' % (value, lower, upper)
        return float(lower), float(upper)

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

        print '---------- Q function -----------'
        ordered = collections.OrderedDict(sorted(self.Q.iteritems(), key=lambda x: x[0], reverse=True))
        prev_key = None
        for key, v in ordered.iteritems():
            if key[0] != prev_key:
                print '\n'
            prev_key = key[0]
            print '%s, %8s : %.3f' % (key[0], self.actions[key[1]], v)
        pass

    def epsilon(self, steps):
        if self.fixed_epsilon:
            return 0.1
        else:
            e = 0.2 / np.log10(steps+10)
            print 'epsilon: %f' % e
            return e
