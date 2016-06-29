from environment import POMDPEnvironment
from gpcontroller_presentation import GPController
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
        return self.belief

    def do_steps(self, n=100):
        for i in range(n):
            episode_end = self.do_step()
            if episode_end:
                self.init_episode()  # reset belief to initial belief [0.65, 0.35]

    def do_step(self):
        print '\nturn: %d' % self.totalTurn
        episode_end = False

        old_belief = new_belief = self.belief
        # old_action = self.best_action
        old_action = self.controller.get_best_action(old_belief)
        action_str = self.get_action_str(old_action)

        if action_str == 'ask':
            # non-terminal step
            observation_num = self.environment.get_observation(old_action)
            new_belief = self.environment.update_belief(
                old_belief, old_action, observation_num)
            pass
        else:
            # terminal step
            episode_end = True
            self.totalEpisode += 1
            pass

        reward = self.environment.observe_reward(old_action)
        new_action = self.controller.get_best_action(new_belief)
        self.controller.observe_step(old_belief, old_action, reward, new_belief, new_action)

        # save belief & action for next turn
        self.belief = new_belief
        self.best_action = new_action
        # counting turn & reward
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

    def test_get_best_action(self):
        self.controller.get_best_action(self.belief)
