from environment import POMDPEnvironment
from op_policy import OptimalPolicy
import numpy as np


class VoiceTask_optimal:

    avg_rewards = []

    def __init__(self, env_file, prior):
        self.environment = POMDPEnvironment(env_file)
        self.prior = self.belief = prior
        self.best_action = np.random.choice(len(self.environment.actions))
        self.totalTurn = 0
        self.totalReward = 0
        self.totalEpisode = 0

        self.controller = OptimalPolicy('examples/policy/voicemail.policy')
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
                self.best_action = self.controller.get_best_action(self.belief)
                avg_reward = float(np.round((self.totalReward / self.totalEpisode), 3))
                print 'avg reward: %.3f' % avg_reward
                self.avg_rewards.append(tuple((self.totalEpisode, avg_reward)))

    def do_step(self):
        print '\nturn: %d' % self.totalTurn
        episode_end = False

        old_belief = self.belief
        old_action = self.controller.get_best_action(old_belief)
        action_str = self.get_action_str(old_action)
        reward = self.environment.observe_reward(old_action)

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
            new_belief = self.belief
            pass

        # save belief & action for next turn
        self.belief = new_belief
        # counting turn & reward
        self.totalTurn += 1
        self.totalReward += reward

        return episode_end

    def do_episodes(self, n=1):
        while True:
            if self.totalEpisode == n:
                break
            episode_end = self.do_step()
            if episode_end:
                self.init_episode()  # reset belief to initial belief [0.65, 0.35]
                avg_reward = float(np.round((self.totalReward / self.totalEpisode), 3))
                print 'avg reward: %.3f' % avg_reward
                self.avg_rewards.append(tuple((self.totalEpisode, avg_reward)))
        pass

    def print_summary(self):
        self.controller.end()
        print 'optimal policy - total steps: %d' % self.totalTurn
        print '\n-------summary-------------'
        print 'Total Episodes: %d' % self.totalEpisode
        print 'Total Rewards: %d' % self.totalReward
        print 'Avg Reward per Episode: %f' % (self.totalReward / self.totalEpisode)
        print '---------------------------'

    def get_reward_data(self):
        return self.avg_rewards

    def save_results(self, filenm):
        import csv
        avg_rewards = self.get_reward_data()
        with open(filenm, 'wb') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['episode', 'avg_reward'])
            writer.writeheader()
            for (episode, avg_reward) in avg_rewards:
                writer.writerow({'episode': episode, 'avg_reward': avg_reward})

    def get_action_str(self, action_num):
        return self.environment.actions[action_num]

    def get_observation_str(self, observation_num):
        return self.environment.observations[observation_num]

    def test_get_best_action(self):
        self.controller.get_best_action(self.belief)


