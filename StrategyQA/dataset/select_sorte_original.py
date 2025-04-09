import ast
import pandas as pd

# 读取CSV文件
file_path = 'question_context_qr_score.csv'  # 请替换为你的文件路径
data = pd.read_csv(file_path)

# 初始化分数统计
grade_counts = {}

# 遍历"Sorted Rewritten Questions"列，解析并统计分数
data['Rewritten Queries with Scores'] = data['Rewritten Queries with Scores'].apply(ast.literal_eval)
for rewritten_questions in data['Rewritten Queries with Scores']:
    for _, score in rewritten_questions:
        grade_counts[score] = grade_counts.get(score, 0) + 1

# 打印统计结果
print("Score Frequency Distribution:")
for score, count in sorted(grade_counts.items()):
    print(f"Score {score}: {count}")

