import pandas as pd
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
import re

ollama_llm = 'qwen2.5:7b'
model = Ollama(
    model=ollama_llm,
    temperature=0.1,
)

# 定义模板
TEMPLATE = """
        In response to the question posed: {question}, please provide a concise and accurate answer below.
        Ensure that your response is directly relevant, extremely succinct, and encapsulates the answer in the fewest words possible.
      """

prompt = PromptTemplate(
    input_variables=["question"],
    template=TEMPLATE
)
llm_chain = prompt | model


def clean_answer(answer):
    """清洗答案内容，去掉非字母数字字符，保留中间内容。"""
    answer = re.sub(r'[^\w\s]', '', answer)  # Remove punctuation
    return answer.strip().lower()


def compute_em(gold_answer, pred_answer):
    """计算 Exact Match (EM) 指标"""
    return int(gold_answer.strip().lower() == pred_answer.strip().lower())


def compute_f1(gold_answer, pred_answer):
    """计算 F1 指标"""
    gold_tokens = gold_answer.strip().lower().split()
    pred_tokens = pred_answer.strip().lower().split()
    common = set(gold_tokens) & set(pred_tokens)
    num_same = len(common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


input_csv_path = 'hotpotqa_test_qr.csv'  # 替换为实际文件路径
df = pd.read_csv(input_csv_path)

y_true = []  # 真实标签
y_pred = []  # 模型预测标签
mismatches = []  # 存储不匹配情况
total = len(df)  # 总问题数


for index, row in tqdm(df.iterrows(), total=len(df)):
    question = row['question']
    correct_answer = row['answer']
    correct_answer = clean_answer(correct_answer)

    result = llm_chain.invoke({"question": question}).replace('"', '').replace('.', '').replace("'", "")
    result = clean_answer(result)

    y_pred.append(result)
    y_true.append(correct_answer)

    if result != correct_answer:
        mismatches.append((correct_answer, result))


# 计算 EM 和 F1 指标
em_scores = [compute_em(ground, pred) for ground, pred in zip(y_true, y_pred)]
f1_scores = [compute_f1(ground, pred) for ground, pred in zip(y_true, y_pred)]

average_em = sum(em_scores) / len(em_scores)
average_f1 = sum(f1_scores) / len(f1_scores)

# 输出结果
print("\nEvaluation Metrics:")
print("Exact Match (EM):", average_em)
print("F1 Score:", average_f1)

# 打印不匹配的情况
print("\nMismatched Cases:")
for mismatch in mismatches:
    print(f"Expected: {mismatch[0]}, Predicted: {mismatch[1]}")
