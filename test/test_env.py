"""
Basic tester for py-pomdp.
"""

# Builtins
import os
import sys
import unittest

from environment import POMDPEnvironment


class POMDPEnvTest(unittest.TestCase):
    """Tests the loading of the POMDP Environment."""

    def setUp(self):
        """Load the POMDP Environment before each test.

        This is a bit excessive, but it's quick and eliminates shared
        state across tests.
        """
        # Load pomdp
        pomdpfile = "../examples/env/voicemail.pomdp"
        self.mypomdp = POMDPEnvironment(pomdpfile)

        # Values
        self.testdiscount = 0.95
        self.testvalues = 'reward'
        self.teststates = ['save', 'delete']
        self.testactions = ['ask', 'doSave', 'doDelete']
        self.testobservations = ['hearSave', 'hearDelete']

    def test_headers(self):
        """Tests that the header values are as expected."""
        self.assertEqual(self.mypomdp.discount, self.testdiscount)
        self.assertEqual(self.mypomdp.values, self.testvalues)
        self.assertEqual(self.mypomdp.states, self.teststates)
        self.assertEqual(self.mypomdp.actions, self.testactions)
        self.assertEqual(self.mypomdp.observations, self.testobservations)

    def test_transitions(self):
        """Check transition values.

        Ask should be identity; all other actions should have the same
        vals.
        """
        for i in range(len(self.testactions)):
            if i == 0:
                # ask action
                for j in range(len(self.teststates)):
                    for k in range(len(self.teststates)):
                        expect = 1 if j == k else 0
                        self.assertEqual(self.mypomdp.T[(i, j, k)], expect)
            else:
                # all other actions
                for j in range(len(self.teststates)):
                    self.assertEqual(self.mypomdp.T[(i, j, 0)], 0.65)
                    self.assertEqual(self.mypomdp.T[(i, j, 1)], 0.35)

    def test_observations(self):
        """Check observation values.

        Ask should have hand-tuned values; others should be uniform.
        """
        for i in range(len(self.testobservations)):
            if i == 0:
                # O: ask
                # 0.8 0.2
                # 0.3 0.7
                self.assertEqual(self.mypomdp.Z[(0, 0, 0)], 0.8)
                self.assertEqual(self.mypomdp.Z[(0, 0, 1)], 0.2)
                self.assertEqual(self.mypomdp.Z[(0, 1, 0)], 0.3)
                self.assertEqual(self.mypomdp.Z[(0, 1, 1)], 0.7)
            else:
                # all other actions
                expect = 1.0 / float(len(self.testobservations))
                for j in range(len(self.teststates)):
                    for k in range(len(self.testobservations)):
                        assert self.mypomdp.Z[(i, j, k)] == expect

    def test_rewards(self):
        """Check the rewards are as expected."""
        # R: ask : * : * : * -1
        for i in range(len(self.teststates)):
            for j in range(len(self.teststates)):
                for k in range(len(self.testobservations)):
                    assert self.mypomdp.R[(0, i, j, k)] == -1

        # R: doSave : save : * : * 5
        # R: doSave : delete : * : * -10
        for j in range(len(self.teststates)):
            for k in range(len(self.testobservations)):
                self.assertEqual(self.mypomdp.R[(1, 0, j, k)], 5)
                self.assertEqual(self.mypomdp.R[(1, 1, j, k)], -10)

        # R: doDelete : save : * : * -20
        # R: doDelete : delete : * : * 5
        for j in range(len(self.teststates)):
            for k in range(len(self.testobservations)):
                self.assertEqual(self.mypomdp.R[(2, 0, j, k)], -20)
                self.assertEqual(self.mypomdp.R[(2, 1, j, k)], 5)

if __name__ == '__main__':
    unittest.main()
