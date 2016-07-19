"""
run it on ipython with following command

ipython --matplotlib qt

"""

import csv
from matplotlib import pyplot as plt
import sys
import numpy as np

avg_rewards = []

files = [
    'rewards_dist_random.csv',
    'rewards_dist_sarsa.csv',
    'rewards_dist_optimal.csv',
    'rewards_dist_gptd.csv',
    'rewards_dist_sarsa_after.csv',
    'rewards_dist_gptd_after.csv',

]

labels = [
    'random',
    'sarsa',
    'optimal',
    'gptd',
    'sarsa_after',
    'gptd_after'
]

colors = [
    'r',
    'g-.',
    'b',
    'm-.',
    'g',
    'm'
]

lines = []

for i in range(len(files)):
    filenm = files[i]
    color = colors[i]

    with open(filenm) as csvfile:
        episodes = []
        rewards = []
        reader = csv.DictReader(csvfile)
        for row in reader:
            episodes.append(row['episode'])
            rewards.append(row['avg_reward'])

        line = plt.plot(episodes, rewards, color, label=labels[i])
        lines.append(line)

# plt.legend(lines, [files])
plt.legend(loc='best')
plt.grid(b=True, which='major', color='b', linestyle='--')

plt.yticks(np.arange(-10.0, 10.0, 2.0))

plt.xlabel('episode', fontsize=18, color='blue')
plt.ylabel('average reward', fontsize=18, color='blue')
plt.show()
