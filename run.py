import numpy as np
# from task import VoiceTask
from task_ql import VoiceTask
# from task_presentation import VoiceTask
import csv

if __name__ == '__main__':

    env_file = 'examples/env/voicemail.pomdp'

    task = VoiceTask(env_file, np.array([[0.65], [0.35]]))

    task.do_steps(3000)
    task.print_summary()

    # write to csv file
    avg_rewards = task.get_reward_data()
    with open('output/rewards_dist.csv', 'wb') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['episode', 'avg_reward'])
        writer.writeheader()
        for (episode, avg_reward) in avg_rewards:
            writer.writerow({'episode': episode, 'avg_reward': avg_reward})
