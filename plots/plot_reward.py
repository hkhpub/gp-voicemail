"""
run it on ipython with following command

ipython --matplotlib qt

"""

import csv
from matplotlib import pyplot as plt
import numpy as np

avg_rewards = []

episodes = []
rewards = []

with open('rewards_dist.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        episodes.append(row['episode'])
        rewards.append(row['avg_reward'])

plt.plot(episodes, rewards, 'r')
plt.grid(b=True, which='major', color='b', linestyle='--')

plt.yticks(np.arange(float(min(rewards)), float(max(rewards))+1, 2.0))

plt.xlabel('episode', fontsize=18, color='blue')
plt.ylabel('average reward', fontsize=18, color='blue')
plt.show()
