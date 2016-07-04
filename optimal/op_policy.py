"""
copyright Kwangho Heo
2016-07-04
"""

# XML parsing
from xml.etree.ElementTree import *

# matrix math
import numpy as np


class OptimalPolicy:

    def __init__(self, filename):
        tree = parse(filename)
        root = tree.getroot()
        avec = list(root)[0]
        alphas = list(avec)
        self.action_nums = []
        val_arrs = []
        for alpha in alphas:
            self.action_nums.append(int(alpha.attrib['action']))
            vals = []
            for val in alpha.text.split():
                vals.append(float(val))
            val_arrs.append(vals)
        self.pMatrix = np.array(val_arrs)

    def get_best_action(self, belief):
        """
        Returns tuple:
            (best-action-num, expected-reward-for-this-action).
        """
        res = self.pMatrix.dot(belief)
        highest_expected_reward = res.max()
        best_action = self.action_nums[res.argmax()]
        # return best_action, highest_expected_reward
        return best_action

    def end(self):
        print 'end debug here'
