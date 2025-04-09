import matplotlib.pyplot as plt
import numpy as np

# 示例数据
k_values = [1, 3, 5, 10]
methods = ['Standard RAG', 'Rewrite-RAG', 'R² (Ours)']
metrics = {
    'nDCG': {
        'top-k': {
            'Standard RAG': [0.60, 0.65, 0.68, 0.70],
            'Rewrite-RAG': [0.65, 0.70, 0.72, 0.74],
            'R² (Ours)': [0.70, 0.75, 0.78, 0.80]
        },
        'bottom-k': {
            'Standard RAG': [0.50, 0.55, 0.58, 0.60],
            'Rewrite-RAG': [0.55, 0.60, 0.63, 0.65],
            'R² (Ours)': [0.60, 0.65, 0.68, 0.70]
        }
    },
    'MAP': {
        'top-k': {
            'Standard RAG': [0.55, 0.60, 0.63, 0.65],
            'Rewrite-RAG': [0.60, 0.65, 0.68, 0.70],
            'R² (Ours)': [0.65, 0.70, 0.73, 0.75]
        },
        'bottom-k': {
            'Standard RAG': [0.45, 0.50, 0.53, 0.55],
            'Rewrite-RAG': [0.50, 0.55, 0.58, 0.60],
            'R² (Ours)': [0.55, 0.60, 0.63, 0.65]
        }
    }
}

# 绘制 nDCG@k 折线图
plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(k_values, metrics['nDCG']['top-k'][method], marker='o', label=f'{method} (top-k)')
    plt.plot(k_values, metrics['nDCG']['bottom-k'][method], marker='s', label=f'{method} (bottom-k)')
plt.xlabel('k')
plt.ylabel('nDCG')
plt.title('nDCG@k Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 绘制 MAP 折线图
plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(k_values, metrics['MAP']['top-k'][method], marker='o', label=f'{method} (top-k)')
    plt.plot(k_values, metrics['MAP']['bottom-k'][method], marker='s', label=f'{method} (bottom-k)')
plt.xlabel('k')
plt.ylabel('MAP')
plt.title('MAP Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 绘制 nDCG@k 柱状图
plt.figure(figsize=(10, 6))
bar_width = 0.2
index = np.arange(len(k_values))
for i, method in enumerate(methods):
    plt.bar(index + i * bar_width, metrics['nDCG']['top-k'][method], bar_width, label=f'{method} (top-k)')
    plt.bar(index + i * bar_width + bar_width * 3, metrics['nDCG']['bottom-k'][method], bar_width, label=f'{method} (bottom-k)')
plt.xlabel('k')
plt.ylabel('nDCG')
plt.title('nDCG@k Performance Comparison')
plt.xticks(index + bar_width, k_values)
plt.legend()
plt.grid(True)
plt.show()

# 绘制 MAP 柱状图
plt.figure(figsize=(10, 6))
bar_width = 0.2
index = np.arange(len(k_values))
for i, method in enumerate(methods):
    plt.bar(index + i * bar_width, metrics['MAP']['top-k'][method], bar_width, label=f'{method} (top-k)')
    plt.bar(index + i * bar_width + bar_width * 3, metrics['MAP']['bottom-k'][method], bar_width, label=f'{method} (bottom-k)')
plt.xlabel('k')
plt.ylabel('MAP')
plt.title('MAP Performance Comparison')
plt.xticks(index + bar_width, k_values)
plt.legend()
plt.grid(True)
plt.show()