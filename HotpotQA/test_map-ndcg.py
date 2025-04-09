import numpy as np
import math


def compute_map_at_k(predicted, ground_truth, k):
    """
    计算 MAP@k
    参数：
        predicted: 预测排序列表，例如 [("q_id", score), ...]
        ground_truth: 黄金排序列表，例如 [("q_id", score), ...]
        k: 截止排名
    返回：
        MAP@k 值
    """
    # 取前 k 个候选项（这里只取候选项标识，不需要得分）
    pred_k = [item[0] for item in predicted[:k]]
    # 取黄金排序 top-k 的候选项作为相关集合
    gt_set = set([item[0] for item in ground_truth[:k]])

    num_relevant = 0
    precision_sum = 0.0

    for i, candidate in enumerate(pred_k, start=1):
        if candidate in gt_set:
            num_relevant += 1
            precision = num_relevant / i
            precision_sum += precision

    # 如果黄金集合中没有任何候选（理论上不可能），返回 0
    if len(gt_set) == 0:
        return 0.0
    return precision_sum / len(gt_set)


def compute_ndcg_at_k(predicted, ground_truth, k):
    """
    计算 nDCG@k
    参数：
        predicted: 预测排序列表，例如 [("q_id", score), ...]
        ground_truth: 黄金排序列表，例如 [("q_id", score), ...]
        k: 截止排名
    返回：
        nDCG@k 值
    """
    # 对预测排序列表取前 k 个候选项，并构造二值相关性：在黄金 top-k 中为1，否则为0
    gt_set = set([item[0] for item in ground_truth[:k]])
    pred_k = [1 if item[0] in gt_set else 0 for item in predicted[:k]]

    dcg = 0.0
    for i, rel in enumerate(pred_k, start=1):
        dcg += rel / math.log2(i + 1)

    # 理想排序：所有 k 个候选项都相关
    idcg = sum([1.0 / math.log2(i + 1) for i in range(1, k + 1)])
    if idcg == 0:
        return 0.0
    return dcg / idcg


# 示例数据（请用实际数据替换）
sorted_answer = [
    ("q1", 0.95),
    ("q2", 0.90),
    ("q3", 0.85),
    ("q4", 0.80),
    ("q5", 0.75)
]

sorted_ranker = [
    ("q2", 0.88),
    ("q1", 0.87),
    ("q4", 0.86),
    ("q5", 0.83),
    ("q3", 0.80)
]

# 设置 top-k 值（例如 k=1 到 5）
k_values = [1, 2, 3, 4, 5]
map_scores = []
ndcg_scores = []

for k in k_values:
    map_k = compute_map_at_k(sorted_ranker, sorted_answer, k)
    ndcg_k = compute_ndcg_at_k(sorted_ranker, sorted_answer, k)
    map_scores.append(map_k)
    ndcg_scores.append(ndcg_k)

print("k\tMAP@k\t\tnDCG@k")
for k, m, n in zip(k_values, map_scores, ndcg_scores):
    print(f"{k}\t{m:.4f}\t\t{n:.4f}")
