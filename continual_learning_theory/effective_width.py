import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. 定义多层感知机 (MLP) 架构（宽度作为变量）
class VariableWidthModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=2000, num_classes=10):
        super(VariableWidthModel, self).__init__()
        # 与论文一致的 3 个隐藏层
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        # 论文要求使用 Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 使用对数尺度 (Log-scale) 选取网络宽度，覆盖从极窄到过参数化
    widths = [4,6,10,16,24,32,48,64,100, 316, 1000, 2000, 3162] # 移除10000以防显存溢出，可自行加回
    
    # 实验超参数
    step_size = 0.003
    num_runs = 3  # 为了绘制带阴影的标准误(Standard Error)，每个宽度运行多次 (论文中是30次)
    
    # 加载 MNIST 数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x)) 
    ])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                          download=True, transform=transform)
    
    # batch_size=1 以严格模拟完全的在线持续学习
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, 
                                              shuffle=True, num_workers=2)

    results = []

    print("开始建立 '有效宽度-准确率' 基线 (Baseline)...\n")

    for width in widths:
        print(f"========== 正在测试隐藏层宽度: {width} ==========")
        run_accuracies = []
        
        for run in range(num_runs):
            model = VariableWidthModel(hidden_size=width).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=step_size) 
            
            # 每个 Run 使用全新的随机排列
            permutation = torch.randperm(28 * 28).to(device)
            
            correct = 0
            total = 0
            
            model.train()
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 应用排列打乱像素
                inputs = inputs[:, permutation]
                
                # 预测和损失计算
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 在线参数更新
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            online_accuracy = 100 * correct / total
            run_accuracies.append(online_accuracy)
            print(f"  Run {run+1}/{num_runs} - 在线准确率: {online_accuracy:.2f}%")
            
        # 计算多次运行的均值和标准误
        mean_acc = np.mean(run_accuracies)
        std_err = np.std(run_accuracies) / np.sqrt(num_runs) if num_runs > 1 else 0.0
        
        print(f"--> 宽度 {width} 总结: 均值 {mean_acc:.2f}%, 标准误 {std_err:.2f}%\n")
        
        results.append({
            'Width': width,
            'Online_Accuracy': mean_acc,
            'Std_Error': std_err
        })

    # ==========================================
    # 1. 保存 CSV 数据
    # ==========================================
    df = pd.DataFrame(results)
    csv_filename = "width_accuracy_baseline.csv"
    df.to_csv(csv_filename, index=False)
    print(f"数据收集完成，已保存至：{csv_filename}")
    
    # ==========================================
    # 2. 论文格式制图 (Nature Style)
    # ==========================================
    print("正在生成标准学术格式曲线图...")
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('white')

    # 绘制均值实线与标准误阴影区域
    ax.plot(df['Width'], df['Online_Accuracy'], 
            color='#1f77b4', linewidth=1.5, label='Network trained from scratch')
    
    if num_runs > 1:
        ax.fill_between(df['Width'], 
                        df['Online_Accuracy'] - df['Std_Error'], 
                        df['Online_Accuracy'] + df['Std_Error'], 
                        color='#1f77b4', alpha=0.2)

    # X轴设置为对数坐标
    ax.set_xscale('log')

    # 设置四周细边框
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')

    # 刻度线格式：四周向内的 tick marks
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, length=4, direction='in', top=True, right=True)
    ax.tick_params(axis='x', which='minor', width=0.6, length=2, direction='in', top=True)

    # 坐标轴标签
    ax.set_xlabel('Network width', fontsize=11, color='black')
    ax.set_ylabel('Online classification accuracy (%)', fontsize=11, color='black')

    # 动态设置 Y 轴范围，留出上下边距
    min_acc = max(0, df['Online_Accuracy'].min() - 5)
    max_acc = min(100, df['Online_Accuracy'].max() + 2)
    ax.set_ylim(min_acc, max_acc)
    ax.set_xlim(8, max(widths) * 1.2)

    # 手动指定重要的宽度标签，关闭科学计数法
    ticks_to_show = [w for w in [10, 100, 1000, 10000] if w <= max(widths) * 10]
    ax.set_xticks(ticks_to_show)
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    # 保存图片
    plot_filename = 'width_accuracy_baseline_plot.png'
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图表已生成并保存至：{plot_filename}")

if __name__ == '__main__':
    main()