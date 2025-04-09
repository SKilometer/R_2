import matplotlib.pyplot as plt

# 数据
K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BoolQ = [0.7164, 0.765, 0.805, 0.8675, 0.845, 0.826, 0.792, 0.762, 0.754, 0.734]
HotpotQA = [0.285, 0.367, 0.4345, 0.485, 0.526, 0.493, 0.465, 0.457, 0.46, 0.438]
MedMCQA = [0.682, 0.7225, 0.7775, 0.839, 0.8645, 0.892, 0.865, 0.854, 0.812, 0.8]
StrategyQA = [0.784, 0.8375, 0.876, 0.905, 0.937, 0.908, 0.885, 0.836, 0.824, 0.815]

# 设置画布大小
plt.figure(figsize=(6, 4))

# 绘制折线图
plt.plot(K, BoolQ, marker='o', label='BoolQ')
plt.plot(K, HotpotQA, marker='^', label='HotpotQA')
plt.plot(K, MedMCQA, marker='s', label='MedMCQA')
plt.plot(K, StrategyQA, marker='D', label='StrategyQA')

# 坐标轴与标题
plt.xlabel('Number of Query Rewrites (k)')
plt.ylabel('Accuracy')
plt.title('Performance on Different ODQA Datasets')

# 显示图例和网格
plt.legend(loc='best')
plt.grid(True)

# 调整布局并保存
plt.tight_layout()
plt.savefig('odqa_performance_line_chart.png', dpi=300)
plt.show()
