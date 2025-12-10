import matplotlib.pyplot as plt
import numpy as np

# Data for V100 and A100
tasks = ['ResNet-50', 'DenseNet-161', 'VGG19', 'Transformer']
v100_execution_time = [2.24, 1.81, 2.36, 2.31]
a100_execution_time = [1, 1, 1, 1]

# Normalize by A100 (divide each value by A100 execution time which is 1)
v100_normalized = np.array(v100_execution_time) / 1
a100_normalized = np.array(a100_execution_time) / 1

# Set bar positions
x = np.arange(len(tasks))  # Task labels

# Bar width
bar_width = 0.35

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# 调换位置：先绘制 A100（左侧），再绘制 V100（右侧）
bars_a100 = ax.bar(x - bar_width / 2, a100_normalized, bar_width, label='A100', color='green')
bars_v100 = ax.bar(x + bar_width / 2, v100_normalized, bar_width, label='V100', color='orange')

# Adding labels and title with fontsize 14
ax.set_xlabel('DL Jobs', fontsize=14)
ax.set_ylabel('Normalized Job Execution Time w.r.t. A100', fontsize=14)
ax.set_title('Normalized Job Execution Time Comparison between V100 and A100', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=14)
ax.legend(fontsize=14)

# Adding the execution time labels on top of the bars
for bars in [bars_a100, bars_v100]:  # 更新遍历顺序以匹配新的绘制顺序
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom', fontsize=14)

plt.tight_layout()
plt.savefig('GPU-gap-2.png', dpi=300, bbox_inches='tight')
# plt.show()
