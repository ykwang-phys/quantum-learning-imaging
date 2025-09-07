# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 17:12:28 2025

@author: 27432
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Load the saved results
with open("success_results.pkl", "rb") as f:
    success_dict = pickle.load(f)

# Plotting
fig, ax = plt.subplots()

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

C = success_dict[next(iter(success_dict))].shape[1]
sam_nums = list(success_dict.keys())
colors = ['C0', 'C1', 'C2']

for idx, (sam_num, linestyle) in enumerate(zip(sam_nums, ['-', '--', '-.'])):
    data = success_dict[sam_num]
    mean_success = data.mean(axis=0)
    min_success = data.min(axis=0)
    max_success = data.max(axis=0)

    x_vals = np.arange(1, C + 1)
    ax.set_xlim(0, C + 1)  # Ensures 0 is visible and plot isn't cut off
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.plot(x_vals, mean_success, linestyle=linestyle, label=f'{sam_num:.0e}', color=colors[idx])
    ax.fill_between(x_vals, min_success, max_success, alpha=0.2, color=colors[idx])

plt.legend(fontsize=17, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)
plt.savefig('P_success_with_range_from_saved.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.show()
