# -*- coding: utf-8 -*-
import numpy as np
import random
import collections


class QLController:

    # belief vector quantize size e.g. (0~0.05) -> 0,      (0.05001 ~ 0.1)
    # 20 x 20
    grid_size = 0.05
    gamma = 0.9
    # learning rate
    alpha = 0.5

    # learning step count
    steps = 0

    def __init__(self, states, actions, initial_belief, initial_action, fixed_epsilon=True):
        self.fixed_epsilon = fixed_epsilon

        self.states = states
        self.actions = actions

        self.B = [(initial_belief, initial_action)]
        self.Q = collections.OrderedDict()
        pass

    def get_best_action(self, belief):
        if np.random.random() <= self.epsilon(self.steps):
            print '>>> epsilon greedy action'
            best = self.get_random_action()
        else:
            belief_tup = self.quantize_belief(belief)
            best = self.getMaxQ(belief_tup)[1]
            print 'maxQ action: %s' % self.actions[best]
        return best

    def get_random_action(self):
        return np.random.choice(len(self.actions))

    def observe_step(self, old_belief, action, reward, new_belief, non_terminal=False):
        self.steps += 1
        print 'old_belief: %s' % np.round(old_belief.flatten(), 3)
        print 'action: %s' % self.actions[action]
        print 'new_belief: %s' % np.round(new_belief.flatten(), 3)
        print 'reward: %s' % reward

        # Q(S, A) + alpha[R + gamma*maxQ(S', a) - Q(S, A)]
        old_belief_tup = self.quantize_belief(old_belief)
        new_belief_tup = self.quantize_belief(new_belief)

        maxQ = self.getMaxQ(new_belief_tup)[0]
        q_value = self.getQ(old_belief_tup, action)
        new_q = q_value + self.alpha * (reward + self.gamma * maxQ) - q_value
        self.Q[(old_belief_tup, action)] = new_q
        pass

    def getMaxQ(self, belief_tup):
        max = -10000
        max_action = 0
        for action in range(len(self.actions)):
            if self.getQ(belief_tup, action) > max:
                max = self.getQ(belief_tup, action)
                max_action = action
        return max, max_action

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
        upper = np.round(self.grid_size*(section), 3)
        # print '%.2f\'s grid is: [%f~%f)' % (value, lower, upper)
        return float(lower), float(upper)

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
