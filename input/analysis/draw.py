import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 读取数据
# 获取项目根目录
project_root = Path(__file__).parent.parent  # 假设脚本在项目子目录中
file_path = project_root / "segments_base.csv"
df = pd.read_csv(file_path)

# 创建图表 - 分开显示CPU和GPU
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Task Segment Analysis (CPU and GPU Separated)', fontsize=16)

# 1. CPU TFLOP分布 (仅包含C_TFLOP > 0的数据)
df_cpu = df[df['C_TFLOP'] > 0]
axes[0, 0].hist(df_cpu['C_TFLOP'], bins=30, alpha=0.7, label='CPU TFLOP', color='blue')
axes[0, 0].set_xlabel('CPU TFLOP')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of CPU TFLOP (C_TFLOP > 0)')
axes[0, 0].legend()

# 2. GPU TFLOP分布 (仅包含G_TFLOP > 0的数据)
df_gpu = df[df['G_TFLOP'] > 0]
axes[0, 1].hist(df_gpu['G_TFLOP'], bins=30, alpha=0.7, label='GPU TFLOP', color='orange')
axes[0, 1].set_xlabel('GPU TFLOP')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of GPU TFLOP (G_TFLOP > 0)')
axes[0, 1].legend()

# 3. 按任务类型分组的平均CPU TFLOP (仅包含C_TFLOP > 0的数据)
task_avg_cpu = df_cpu.groupby('template_id')['C_TFLOP'].mean()
axes[1, 0].bar(range(len(task_avg_cpu)), task_avg_cpu.values, color='blue', alpha=0.7)
axes[1, 0].set_xlabel('Template ID')
axes[1, 0].set_ylabel('Average CPU TFLOP')
axes[1, 0].set_title('Average CPU TFLOP by Template ID (C_TFLOP > 0)')
axes[1, 0].set_xticks(range(len(task_avg_cpu)))
axes[1, 0].set_xticklabels(task_avg_cpu.index, rotation=45)

# 4. 按任务类型分组的平均GPU TFLOP (仅包含G_TFLOP > 0的数据)
task_avg_gpu = df_gpu.groupby('template_id')['G_TFLOP'].mean()
axes[1, 1].bar(range(len(task_avg_gpu)), task_avg_gpu.values, color='orange', alpha=0.7)
axes[1, 1].set_xlabel('Template ID')
axes[1, 1].set_ylabel('Average GPU TFLOP')
axes[1, 1].set_title('Average GPU TFLOP by Template ID (G_TFLOP > 0)')
axes[1, 1].set_xticks(range(len(task_avg_gpu)))
axes[1, 1].set_xticklabels(task_avg_gpu.index, rotation=45)

plt.tight_layout()
plt.savefig('task_analysis_separated.png', dpi=300, bbox_inches='tight')
plt.show()
