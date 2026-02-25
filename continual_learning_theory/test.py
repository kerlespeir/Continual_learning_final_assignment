import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. 定义多层感知机 (MLP) 架构
class FixedWidthModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=2000, num_classes=10):
        super(FixedWidthModel, self).__init__()
        # 论文标准设置: 3个隐藏层，每层2000个单元
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

    # 实验超参数
    hidden_size = 100
    num_tasks = 800        # 测试的新任务数量（论文中是 800，这里用 50 演示可塑性丧失趋势）
    step_size = 0.003     # 论文中推荐的 SGD 较优步长
    
    # 加载 MNIST 数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x)) 
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                          download=True, transform=transform)
    
    # 纯在线学习，batch_size=1
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, 
                                              shuffle=True, num_workers=2)

    print(f"开始持续学习测试 (网络宽度: {hidden_size}, 连续任务数: {num_tasks})...\n")

    # 注意：模型只在第1个任务开始前初始化一次！后续任务绝不重新初始化。
    model = FixedWidthModel(hidden_size=hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=step_size) 
    
    task_accuracies = []

    for task in range(num_tasks):
        # 针对当前新任务，生成一个全新的随机像素排列 (代表数据分布的剧烈改变)
        permutation = torch.randperm(28 * 28).to(device)
        
        correct = 0
        total = 0
        
        model.train()
        # 数据流 (Stream of data)：在线学习当前任务的 60000 个样本
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 应用当前任务的像素排列
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
            
        # 计算当前任务的整体在线准确率
        online_accuracy = 100 * correct / total
        task_accuracies.append(online_accuracy)
        
        print(f"Task {task + 1:<3} / {num_tasks} | 在线准确率: {online_accuracy:.2f}%")

    # ==========================================
    # 1. 保存 CSV 数据
    # ==========================================
    df = pd.DataFrame({
        'Task_Number': range(1, num_tasks + 1),
        'Online_Accuracy': task_accuracies
    })
    csv_filename = "continual_learning_plasticity_loss.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n持续学习测试完成，结果已保存至：{csv_filename}")
    
    # ==========================================
    # 2. 论文格式制图 (Nature Style)
    # ==========================================
    print("正在生成持续学习可塑性丧失曲线图...")
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('white')

    # 绘制准确率下降曲线 (由于单次 run 演示，没有画阴影)
    ax.plot(df['Task_Number'], df['Online_Accuracy'], 
            color='#d62728', linewidth=1.5, label=f'Backpropagation (Step size={step_size})')

    # 设置四周细边框
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')

    # 刻度线格式：四周向内的 tick marks
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, length=4, direction='in', top=True, right=True)

    # 坐标轴标签
    ax.set_xlabel('Task number', fontsize=11, color='black')
    ax.set_ylabel('Percent correct on MNIST', fontsize=11, color='black')

    # 设置 Y 轴范围 (聚焦于下降趋势区间)
    ax.set_ylim(88, 96)
    ax.set_xlim(0, num_tasks + 1)

    # 添加图例
    ax.legend(frameon=False, loc='upper right', fontsize=10)

    # 保存图片
    plot_filename = 'continual_learning_plasticity_loss.png'
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图表已生成并保存至：{plot_filename}")

if __name__ == '__main__':
    main()