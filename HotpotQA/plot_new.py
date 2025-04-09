import numpy as np
import matplotlib.pyplot as plt

# 颜色定义（科研风格）
colors = {
    'topk': '#4C72B0',      # 深蓝
    'worsek': '#C44E52',    # 深红
    'randomk': '#55A868',   # 深绿
    'rag': '#DD8452',       # 深橙
    'rewriter5': '#8172B2', # 紫色
    'rewriter10': '#937860' # 棕色
}

# 数据准备
K_values = np.array([1, 2, 3, 4, 5])

top_k_acc = np.array([0.80, 0.82, 0.83, 0.85, 0.84])
worse_k_acc = np.array([0.75, 0.78, 0.79, 0.80, 0.82])

random_k_acc_repeats = np.array([
    [0.78, 0.79, 0.81, 0.81, 0.82],
    [0.77, 0.80, 0.80, 0.82, 0.83],
    [0.79, 0.78, 0.82, 0.80, 0.82],
    [0.74, 0.80, 0.79, 0.81, 0.81],
    [0.76, 0.78, 0.81, 0.82, 0.82],
    [0.78, 0.80, 0.82, 0.83, 0.84],
    [0.77, 0.79, 0.81, 0.82, 0.82],
    [0.78, 0.81, 0.80, 0.84, 0.85],
    [0.75, 0.79, 0.80, 0.82, 0.83],
    [0.76, 0.80, 0.81, 0.82, 0.84],
])
random_means = random_k_acc_repeats.mean(axis=0)
random_stds = random_k_acc_repeats.std(axis=0)

# 三条参考水平线
rag_acc = 0.78
rewriter5_acc = 0.82
rewriter10_acc = 0.84

# 开始绘图
plt.figure(figsize=(7.5, 5))

# Top-k
plt.plot(K_values, top_k_acc, label='Top-k', color=colors['topk'],
         marker='o', linewidth=2)

# Worse-k
plt.plot(K_values, worse_k_acc, label='Worse-k', color=colors['worsek'],
         marker='s', linewidth=2)

# Random-k
plt.plot(K_values, random_means, label='Random-k (mean)',
         color=colors['randomk'], marker='^', linewidth=2)
plt.fill_between(K_values,
                 random_means - random_stds,
                 random_means + random_stds,
                 color=colors['randomk'], alpha=0.15,
                 label='Random-k ± std')

# Baseline horizontal lines
plt.axhline(y=rag_acc, color=colors['rag'], linestyle='--',
            linewidth=1.8, label='RAG')
plt.axhline(y=rewriter5_acc, color=colors['rewriter5'], linestyle='--',
            linewidth=1.8, label='Rewriter-5')
plt.axhline(y=rewriter10_acc, color=colors['rewriter10'], linestyle='--',
            linewidth=1.8, label='Rewriter-10')

# 轴标签和图标题
plt.xlabel('K', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Effect of K on Rewrite Selection Strategies', fontsize=14)

# 网格美化
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

# 图例
plt.legend(fontsize=10, loc='lower right', frameon=False)

# 布局优化
plt.tight_layout()
plt.show()
