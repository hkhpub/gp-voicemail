import numpy as np
import csv
# from qlearn.task_ql import VoiceTask_ql
from sarsa.task_sarsa import VoiceTask_sarsa
from gptd.task_gptd import VoiceTask_gptd
from optimal.task_optimal import VoiceTask_optimal
from rand.task_random import VoiceTask_random


if __name__ == '__main__':

    env_file = 'examples/env/voicemail.pomdp'

    # fixed_epsilon = True
    fixed_epsilon = False

    if fixed_epsilon:
        result_files = [
            'plots/rewards_dist_random.csv',
            'plots/rewards_dist_optimal.csv',
            'plots/rewards_dist_gptd_fixed.csv',
            'plots/rewards_dist_sarsa_fixed.csv'
        ]
    else:
        result_files = [
            'plots/rewards_dist_random.csv',
            'plots/rewards_dist_optimal.csv',
            'plots/rewards_dist_gptd.csv',
            'plots/rewards_dist_sarsa.csv'
        ]

    task_list = [
        VoiceTask_random(env_file, np.array([[0.65], [0.35]])),
        VoiceTask_optimal(env_file, np.array([[0.65], [0.35]])),
        VoiceTask_gptd(env_file, np.array([[0.65], [0.35]]), fixed_epsilon),
        VoiceTask_sarsa(env_file, np.array([[0.65], [0.35]]), fixed_epsilon)
    ]

    for i, task in enumerate(task_list):
        # task.do_steps(500)
        task.do_episodes(200)
        task.print_summary()

        # write to csv file
        task.save_results(result_files[i])
