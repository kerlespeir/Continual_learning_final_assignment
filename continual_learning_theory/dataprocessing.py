import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker

def main():
    # ==========================================
    # 1. 加载并构建基线插值函数 (准确率 -> 有效宽度)
    # ==========================================
    try:
        df_baseline = pd.read_csv('width_accuracy_baseline.csv')
    except FileNotFoundError:
        print("请确保 width_accuracy_baseline.csv 在当前目录下！")
        return

    # 排序并去重，以确保插值函数的单调性与合法性
    df_baseline = df_baseline.sort_values(by='Online_Accuracy').drop_duplicates(subset=['Online_Accuracy'])
    baseline_acc = df_baseline['Online_Accuracy'].values
    baseline_log_width = np.log10(df_baseline['Width'].values)

    # 创建插值函数 (线性外推，以应对可能的极值情况)
    f_acc_to_logwidth = interp1d(baseline_acc, baseline_log_width, 
                                 kind='linear', fill_value="extrapolate")

    # ==========================================
    # 2. 读取 WebPlotDigitizer 提取的持续学习数据
    # ==========================================
    try:
        # 读取 CSV (第一行是表头，第二行是 'X', 'Y' 字符串标签，跳过它)
        df_wpd = pd.read_csv('wpd_datasets.csv')
        # 去掉第一行包含 'X' 'Y' 字符的数据，并将剩余数据转为 float
        df_wpd = df_wpd.iloc[1:].astype(float)
    except FileNotFoundError:
        print("请确保 wpd_datasets (1).csv 在当前目录下！")
        return

    # 定义从提取的 CSV 中读取各列数据的规则
    # Dataset 0 -> 宽度 100, Dataset 1 -> 宽度 1000(1k), Dataset 2 -> 宽度 10000(10k)
    data_configs = [
        {'name': 'Width 100',  'col_x': 0, 'col_y': 1, 'color': '#ff7f0e', 'physical_width': 100},
        {'name': 'Width 1k',   'col_x': 2, 'col_y': 3, 'color': '#d62728', 'physical_width': 1000},
        {'name': 'Width 10k',  'col_x': 4, 'col_y': 5, 'color': '#7f7f7f', 'physical_width': 10000},
    ]

    # ==========================================
    # 3. 核心计算与可视化
    # ==========================================
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')

    for config in data_configs:
        # 提取非空数据点 (WebPlotDigitizer 中各数据集的点数可能不同)
        x_vals = df_wpd.iloc[:, config['col_x']].dropna().values
        y_vals = df_wpd.iloc[:, config['col_y']].dropna().values
        
        # 确保数据点按 Task 数 (X 轴) 正序排列
        sort_idx = np.argsort(x_vals)
        x_vals = x_vals[sort_idx]
        y_vals = y_vals[sort_idx]
        
        # --- 将老化准确率转化为有效宽度 ---
        eff_log_widths = f_acc_to_logwidth(y_vals)
        eff_widths = 10 ** eff_log_widths
        
        # 绘制该网络有效宽度的退化曲线
        ax.plot(x_vals, eff_widths, color=config['color'], linewidth=2, label=f"{config['name']} (Effective)")
        
        # 绘制一条半透明的水平虚线代表其初始绝对物理宽度 (理想上限)
        ax.axhline(config['physical_width'], color=config['color'], linestyle='--', linewidth=1.2, alpha=0.5)

    # ==========================================
    # 4. 图表样式微调 (Nature Style)
    # ==========================================
    # 由于宽度的萎缩是指数级的，Y轴必须使用对数坐标
    ax.set_yscale('log')
    
    # 学术风四周边框
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')
        
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, length=4, direction='in', top=True, right=True)
    
    ax.set_xlabel('Task number', fontsize=12, color='black')
    ax.set_ylabel('Effective Network Width', fontsize=12, color='black')
    
    # 格式化Y轴，去除科学计数法，方便阅读
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    
    # 设置合理的坐标范围与图例
    ax.set_xlim(0, 150) # 根据论文图表，网络宽度对比图跑了 150 个 Tasks
    ax.legend(frameon=False, fontsize=10, loc='lower left')
    
    plt.tight_layout()
    plt.savefig('effective_width_comparison.png', dpi=300, bbox_inches='tight')
    print("对比图表已成功生成并保存为：effective_width_comparison.png")

if __name__ == "__main__":
    main()