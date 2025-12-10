import matplotlib.pyplot as plt
import numpy as np

# Data for the tasks (J1, J2, J3, J4, Avg-JCT)
tasks = ['J1', 'J2', 'J3', 'J4', 'Avg-JCT']
gpu_proportional = [6200, 3800, 3700, 2800, 4100]  # GPU-Proportional task times
resource_sensitive = [2800, 1800, 3700, 2800, 2700]  # Resource-Sensitive task times

# Set the x-axis positions
x = np.arange(len(tasks))

# Bar width
bar_width = 0.35

# Plotting
fig, ax = plt.subplots(figsize=(10, 4))

# Bars for GPU-Proportional and Resource-Sensitive
bars_gpu = ax.bar(x - bar_width / 2, gpu_proportional, bar_width, label='GPU-Proportional', hatch='/', color='orange')
bars_resource = ax.bar(x + bar_width / 2, resource_sensitive, bar_width, label='Resource-Sensitive', color='black')

# Adding labels and title
# ax.set_xlabel('Jobs')
ax.set_ylabel('Job Completion Time (ms)',fontsize=14)
# ax.set_title('Comparison of Job Completion Time for GPU-Proportional and Resource-Sensitive Scheduling')
ax.set_xticks(x)
ax.set_xticklabels(tasks,fontsize=14)
ax.legend(fontsize=14)

# Adding the execution time labels on top of the bars
# for bars in [bars_gpu, bars_resource]:
#     for bar in bars:
#         yval = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width() / 2, yval + 100, round(yval, 0), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('quota.png', dpi=300, bbox_inches='tight')
# plt.show()
