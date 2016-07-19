"""
run it on ipython with following command

ipython --matplotlib qt

"""

import csv
from matplotlib import pyplot as plt
import sys
import numpy as np

avg_rewards = []

episodes = []
rewards = []

args = []
if sys.argv is not None:
    args = sys.argv

filenm = args[1] if len(args) >= 2 else 'rewards_dist_optimal.csv'
color = args[2] if len(args) >= 3 else 'g'
print 'loading file >>> ' + filenm

with open(filenm) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        episodes.append(row['episode'])
        rewards.append(row['avg_reward'])

line = plt.plot(episodes, rewards, color)
plt.legend(line, [filenm])

plt.grid(b=True, which='major', color='b', linestyle='--')

plt.yticks(np.arange(-10.0, 10.0, 2.0))

plt.xlabel('episode', fontsize=18, color='blue')
plt.ylabel('average reward', fontsize=18, color='blue')
plt.show()
