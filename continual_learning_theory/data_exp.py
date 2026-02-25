import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker

def main():
    # ==========================================
    # 1. 加载基线数据构建插值函数
    # ==========================================
    try:
        df_baseline = pd.read_csv('width_accuracy_baseline.csv')
    except FileNotFoundError:
        print("请确保 width_accuracy_baseline.csv 在当前目录下！")
        return

    # 排序去重，保证插值函数单调
    df_baseline = df_baseline.sort_values(by='Online_Accuracy').drop_duplicates(subset=['Online_Accuracy'])
    baseline_acc = df_baseline['Online_Accuracy'].values
    baseline_log_width = np.log10(df_baseline['Width'].values)

    # 建立准确率到对数宽度的线性插值映射
    f_acc_to_logwidth = interp1d(baseline_acc, baseline_log_width, 
                                 kind='linear', fill_value="extrapolate")

    # ==========================================
    # 2. 读取 WebPlotDigitizer 持续学习数据
    # ==========================================
    try:
        df_wpd = pd.read_csv('wpd_datasets.csv')
        df_wpd = df_wpd.iloc[1:].astype(float)
    except FileNotFoundError:
        print("请确保 wpd_datasets (1).csv 在当前目录下！")
        return

    # 包含 100, 1k, 和 10k 的网络列配置
    data_configs = [
        {'name': 'Width 100', 'col_x': 0, 'col_y': 1, 'color': '#ff7f0e'},
        {'name': 'Width 1k',  'col_x': 2, 'col_y': 3, 'color': '#d62728'},
        {'name': 'Width 10k', 'col_x': 4, 'col_y': 5, 'color': '#7f7f7f'}  # 新增 10k 网络
    ]

    # ==========================================
    # 3. 绘图与拟合计算
    # ==========================================
    fig, ax = plt.subplots(figsize=(8.5, 6))
    fig.patch.set_facecolor('white')

    print("===== t > 40 有效宽度衰减率分析 =====")
    
    for config in data_configs:
        # 提取数据并清洗排序
        x_vals = df_wpd.iloc[:, config['col_x']].dropna().values
        y_vals = df_wpd.iloc[:, config['col_y']].dropna().values
        
        sort_idx = np.argsort(x_vals)
        x_vals = x_vals[sort_idx]
        y_vals = y_vals[sort_idx]
        
        # 转化为对数有效宽度
        log_eff_widths = f_acc_to_logwidth(y_vals)
        eff_widths = 10 ** log_eff_widths
        
        # 绘制原始数据点 (散点 + 半透明连线)
        ax.plot(x_vals, eff_widths, 'o', markersize=4, color=config['color'], alpha=0.4)
        ax.plot(x_vals, eff_widths, color=config['color'], linewidth=1, alpha=0.3)
        
        # --- 选取 t > 40 的数据进行线性拟合 ---
        mask = x_vals > 40
        x_fit = x_vals[mask]
        log_y_fit = log_eff_widths[mask]
        
        if len(x_fit) > 1:
            # np.polyfit 拟合一次多项式 y = kx + b
            slope, intercept = np.polyfit(x_fit, log_y_fit, 1)
            
            # 计算拟合直线的 Y 值并还原为真实宽度
            fit_line_y = 10 ** (slope * x_fit + intercept)
            
            # 换算百分比衰减率：每增加1个Task，容量衰减的百分比
            decay_pct_per_task = (1 - 10**slope) * 100
            
            print(f"{config['name']} 网络:")
            print(f"  -> 对数斜率 k = {slope:.5f}")
            print(f"  -> 每学习1个新任务，等价容量平均缩水: {decay_pct_per_task:.2f}%")
            
            # 在图表上绘制深色的拟合直线
            ax.plot(x_fit, fit_line_y, color=config['color'], linewidth=2.5, 
                    label=f"{config['name']} Decay rate: {decay_pct_per_task:.2f}")

    # ==========================================
    # 4. 图表学术格式设置
    # ==========================================
    ax.set_yscale('log')
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')
        
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, length=4, direction='in', top=True, right=True)
    
    ax.set_xlabel('Task number', fontsize=12, color='black')
    ax.set_ylabel('Effective Network Width', fontsize=12, color='black')
    ax.set_title('Exponential Decay of Effective Width', fontsize=12, pad=15)
    
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.set_xlim(0, max(x_vals) * 1.05)
    
    # 标出 t=40 的分界线
    ax.axvline(x=40, color='gray', linestyle=':', linewidth=1.5, label='t = 40 (Fit Start)')
    
    # 图例放于左下角以避免遮挡曲线
    ax.legend(frameon=True, edgecolor='black', fontsize=10, loc='lower left')
    
    plt.tight_layout()
    plt.savefig('effective_width_decay_fit.png', dpi=300, bbox_inches='tight')
    print("=====================================")
    print("拟合图表已生成并保存为：effective_width_decay_fit.png")

if __name__ == "__main__":
    main()