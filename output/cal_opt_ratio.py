def calculate_gccs_optimization(input_data):
    """
    针对输入的实验数据，计算 GCCS 相较于其他算法的优化比例。
    输入格式支持多行字符串或列表。
    """
    lines = input_data.strip().split('\n')
    results = {}

    # 1. 解析数据
    for line in lines:
        parts = line.split(',')
        # if len(parts) != 4:
        #     continue

        method = parts[-2].strip()
        makespan = float(parts[-1].strip())
        results[method] = makespan

    # 2. 检查 GCCS 是否在数据中
    if 'GCCS' not in results:
        return "错误：输入数据中未找到 GCCS 的结果。"

    gccs_val = results['GCCS']
    print(f"{'对比算法':<10} | {'Makespan':<12} | {'GCCS 优化幅度 (%)'}")
    print("-" * 45)

    # 3. 计算并输出结果
    for method, val in results.items():
        if method == 'GCCS':
            continue

        # 计算公式：(基准 - GCCS) / 基准 * 100%
        improvement = (val - gccs_val) / val * 100
        print(f"{method:<10} | {val:<12.6f} | {improvement:>8.2f}%")


# --- 使用示例 ---
# 您可以直接把数据粘贴进这个三引号字符串中
raw_data = """
16,1.0R,4,GCCS,0.8882328042645704
16,1.0R,4,HEFT,1.0230797883925075
16,1.0R,4,Hydra,1.0530797883925075
16,1.0R,4,MRSA,1.011211906494032

"""

calculate_gccs_optimization(raw_data)