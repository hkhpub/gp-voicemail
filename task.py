from environment import POMDPEnvironment
from gpcontroller import GPController
import numpy as np


class VoiceTask:

    def __init__(self, env_file, prior):
        self.environment = POMDPEnvironment(env_file)
        self.prior = self.belief = prior
        self.best_action = np.random.choice(len(self.environment.actions))
        self.totalTurn = 0
        self.totalReward = 0
        self.totalEpisode = 0

        self.controller = GPController(self.environment.states,
                                       self.environment.actions,
                                       self.belief, self.best_action)
        self.init_episode()
        pass

    def init_episode(self):
        self.environment.init_episode()
        self.belief = self.prior

    def do_steps(self, n=100):
        for i in range(n):
            episode_end = self.do_step()
            if episode_end:
                self.init_episode()
                self.totalEpisode += 1

    def do_step(self):
        old_belief = new_belief = self.belief
        old_action = self.best_action
        episode_end = False
        # 1. select action
        new_action = self.controller.get_best_action()
        best_action_str = self.get_action_str(new_action)
        if best_action_str == 'ask':
            observation_num = self.environment.get_observation(new_action)
            new_belief = self.environment.update_belief(
                old_belief, new_action, observation_num)
            pass
        else:
            # terminal action: either doSave or doDelete
            episode_end = True
            pass
        reward = self.environment.observe_reward(new_action)
        self.controller.observe_step(old_belief, old_action, reward, new_belief, new_action)
        self.belief = new_belief
        self.best_action = new_action
        self.totalTurn += 1
        self.totalReward += reward

        return episode_end

    def do_episodes(self, n=1):
        pass

    def print_summary(self):
        print '\n-------summary-------------'
        print 'Total Episodes: %d' % self.totalEpisode
        print 'Total Rewards: %d' % self.totalReward
        print 'Avg Reward per Episode: %d' % (self.totalReward / self.totalEpisode)
        print '---------------------------'

    def get_action_str(self, action_num):
        return self.environment.actions[action_num]

    def get_observation_str(self, observation_num):
        return self.environment.observations[observation_num]