import numpy as np
import csv
# from task import VoiceTask
# from qlearn.task_ql import VoiceTask
# from task_presentation import VoiceTask
# from sarsa.task_sarsa import VoiceTask
from gptd.task_gptd import VoiceTask
# from optimal.task_optimal import VoiceTask


if __name__ == '__main__':

    env_file = 'examples/env/voicemail.pomdp'

    task = VoiceTask(env_file, np.array([[0.65], [0.35]]))

    # task.do_steps(5000)
    task.do_episodes(2000)
    task.print_summary()

    # # write to csv file
    avg_rewards = task.get_reward_data()
    with open('plots/rewards_dist_gptd.csv', 'wb') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['episode', 'avg_reward'])
        writer.writeheader()
        for (episode, avg_reward) in avg_rewards:
            writer.writerow({'episode': episode, 'avg_reward': avg_reward})
