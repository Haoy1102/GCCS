import matplotlib.pyplot as plt
import numpy as np

# Data for V100 and RTX 3090 (same structure as the image you provided)
tasks = ['ResNet-50', 'Vgg-19', 'MobileNetV3', 'PPO']
v100_throughput = [6.5, 6.3, 4.7, 1.0]  # throughput for V100
rtx3090_throughput = [8.9, 8.2, 6.8, 1.6]  # throughput for RTX 3090

# Bar width
bar_width = 0.35
# Bar positions
index = np.arange(len(tasks))

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

# Create bars for V100 and RTX 3090
bars_v100 = ax.bar(index, v100_throughput, bar_width, label='V100', color='orange')
bars_rtx3090 = ax.bar(index + bar_width, rtx3090_throughput, bar_width, label='RTX 3090', color='green')

# Adding labels, title, and custom x-axis tick labels
ax.set_xlabel('Tasks')
ax.set_ylabel('Throughput (w.r.t. K80)')
ax.set_title('Throughput Comparison between V100 and RTX 3090')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(tasks)
ax.legend()

# Adding the throughput values on top of each bar
for bars in [bars_v100, bars_rtx3090]:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval , round(yval, 1), ha='center', va='bottom')

# Show plot
plt.tight_layout()
plt.savefig('GPU-gap.png', dpi=300, bbox_inches='tight')
# plt.show()
