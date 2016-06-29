# matrix math
import numpy as np


class POMDPEnvironment:
    """
    true_state: the true user goal for each episode
    1) save: the user actually want to save the message
    2) delete: the user actually want to delete the message
    """
    true_state = None

    def __init__(self, filename):
        """
        Parses .pomdp file and loads info into this object's fields.
        Attributes:
            discount
            values
            states
            actions
            observations
            T
            Z
            R
        """
        f = open(filename, 'r')
        self.contents = [
            x.strip() for x in f.readlines()
            if (not (x.startswith("#") or x.isspace()))
        ]

        # set up transition function T, observation function Z, and
        # reward R
        self.T = {}
        self.Z = {}
        self.R = {}

        # go through line by line
        i = 0
        while i < len(self.contents):
            line = self.contents[i]
            if line.startswith('discount'):
                i = self.__get_discount(i)
            elif line.startswith('values'):
                i = self.__get_value(i)
            elif line.startswith('states'):
                i = self.__get_states(i)
            elif line.startswith('actions'):
                i = self.__get_actions(i)
            elif line.startswith('observations'):
                i = self.__get_observations(i)
            elif line.startswith('T'):
                i = self.__get_transition(i)
            elif line.startswith('O'):
                i = self.__get_observation(i)
            elif line.startswith('R'):
                i = self.__get_reward(i)
            else:
                raise Exception("Unrecognized line: " + line)

        f.close()

    def __get_discount(self, i):
        line = self.contents[i]
        self.discount = float(line.split()[1])
        return i + 1

    def __get_value(self, i):
        # Currently just supports "values: reward". I.e. currently
        # meaningless.
        line = self.contents[i]
        self.values = line.split()[1]
        return i + 1

    def __get_states(self, i):
        # TODO support states as number
        line = self.contents[i]
        self.states = line.split()[1:]
        return i + 1

    def __get_actions(self, i):
        line = self.contents[i]
        self.actions = line.split()[1:]
        return i + 1

    def __get_observations(self, i):
        line = self.contents[i]
        self.observations = line.split()[1:]
        return i + 1

    def __get_transition(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = self.actions.index(pieces[0])

        next_line = self.contents[i+1]
        if next_line == "identity":
            # case 4: T: <action>
            # identity
            for j in range(len(self.states)):
                for k in range(len(self.states)):
                    prob = 1.0 if j == k else 0.0
                    self.T[(action, j, k)] = prob
            return i + 2
        elif next_line == "uniform":
            # case 5: T: <action>
            # uniform
            prob = 1.0 / float(len(self.states))
            for j in range(len(self.states)):
                for k in range(len(self.states)):
                    self.T[(action, j, k)] = prob
            return i + 2
        else:
            # case 6: T: <action>
            # %f %f ... %f
            # %f %f ... %f
            # ...
            # %f %f ... %f
            for j in range(len(self.states)):
                probs = next_line.split()
                assert len(probs) == len(self.states)
                for k in range(len(probs)):
                    prob = float(probs[k])
                    self.T[(action, j, k)] = prob
                next_line = self.contents[i+2+j]
            return i+1+len(self.states)

    def __get_observation(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = self.actions.index(pieces[0])

        next_line = self.contents[i+1]
        if next_line == "identity":
            # case 4: O: <action>
            # identity
            for j in range(len(self.states)):
                for k in range(len(self.observations)):
                    prob = 1.0 if j == k else 0.0
                    self.Z[(action, j, k)] = prob
            return i + 2
        elif next_line == "uniform":
            # case 5: O: <action>
            # uniform
            prob = 1.0 / float(len(self.observations))
            for j in range(len(self.states)):
                for k in range(len(self.observations)):
                    self.Z[(action, j, k)] = prob
            return i + 2
        else:
            # case 6: O: <action>
            # %f %f ... %f
            # %f %f ... %f
            # ...
            # %f %f ... %f
            for j in range(len(self.states)):
                probs = next_line.split()
                assert len(probs) == len(self.observations)
                for k in range(len(probs)):
                    prob = float(probs[k])
                    self.Z[(action, j, k)] = prob
                next_line = self.contents[i+2+j]
            return i + 1 + len(self.states)

    def __get_reward(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = self.actions.index(pieces[0])
        state = self.states.index(pieces[1])
        reward = pieces[2]
        self.R[(action, state)] = reward
        return i + 1

    def get_observation(self, action_num):
        """
        returns an observation based on
        1) true state, 2) action, 3) observation prob.
        it actually simulate speech recognition error
        """
        true_state_num = self.states.index(self.true_state)
        prob_dist = [self.Z[(action_num, true_state_num, i)]
                     for i in range(len(self.observations))]
        observation = np.random.choice(len(self.observations), 1, p=prob_dist)[0]
        # print 'prob_dist: %s' % prob_dist
        # print 'observation: %s' % self.observations[observation]
        return observation

    def observe_reward(self, action_num):
        return float(self.R[(action_num, self.states.index(self.true_state))])
        pass

    def init_episode(self):
        self.true_state = np.random.choice(self.states, 1, p=[0.65, 0.35])[0]
        print '\n(new episode) true state is: %s' % self.true_state
        pass

    def update_belief(self, prev_belief, action_num, observation_num):
        """
        Note that a POMDPEnvironment doesn't hold beliefs, so this takes
        and returns a belief vector.

        prev_belief     numpy array
        action_num      int
        observation_num int
        return          numpy array
        """
        b_new_nonnormalized = []
        for s_prime in range(len(self.states)):
            p_o_prime = self.Z[(action_num, s_prime, observation_num)]
            summation = 0.0
            for s in range(len(self.states)):
                p_s_prime = self.T[(action_num, s, s_prime)]
                b_s = float(prev_belief[s])
                summation += p_s_prime * b_s
            b_new_nonnormalized.append(p_o_prime * summation)

        # normalize
        b_new = []
        total = sum(b_new_nonnormalized)
        for b_s in b_new_nonnormalized:
            b_new.append([b_s/total])
        return np.array(b_new)

    def print_summary(self):
        print "discount:", self.discount
        print "values:", self.values
        print "states:", self.states
        print "actions:", self.actions
        print "observations:", self.observations
        print ""
        print "T:", self.T
        print ""
        print "Z:", self.Z
        print ""
        print "R:", self.R
