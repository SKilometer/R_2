import numpy as np
import matplotlib
import matplotlib.pyplot as plt

custom_rc_params = {
    'figure.figsize': (6, 4),
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'grid.linestyle': ':',
    'grid.alpha': 0.7,
    'axes.grid': True,
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
}

with matplotlib.rc_context(custom_rc_params):
    # 1. 数据准备
    k_values = np.array([1, 2, 3, 4, 5])
    acc_top = np.array([0.81, 0.8225, 0.825, 0.8375, 0.825])
    acc_worse = np.array([0.75, 0.7825, 0.795, 0.8075, 0.825])
    acc_random = np.array([0.7875, 0.8125, 0.8125, 0.82, 0.825])

    rag_baseline = 0.79
    qr_rag_baseline = 0.825

    plt.figure(figsize=(6, 4))

    # 2. 绘制三条曲线
    plt.plot(k_values, acc_top, marker='o', label='Top‑k')
    plt.plot(k_values, acc_worse, marker='s', label='Worse‑k')
    plt.plot(k_values, acc_random, marker='^', label='Random‑k')

    # 3. 绘制两条基线
    plt.axhline(y=rag_baseline, color='red', linestyle='--', label='RAG')
    plt.axhline(y=qr_rag_baseline, color='blue', linestyle='--', label='Rewriter-RAG')

    # 4. 美化与注释
    plt.xlabel('K', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    # plt.title('Comparison of Different Query Selection Strategies on Qwen2.5:32B', fontsize=12)
    plt.ylim([0.70, 0.86])
    plt.xticks(k_values)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    # 5. 保存图像到当前目录（PNG格式）
    plt.savefig("medmcqa.png", dpi=300, bbox_inches='tight')

    # 6. 显示图像（可选）
    plt.show()