import pandas as pd
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import numpy as np

ollama_llm = 'qwen2.5:7b'
model = Ollama(model=ollama_llm)

# 定义模板
TEMPLATE = """
    Based on the provided information, determine whether the following question is true or false:  
    {question}  
    Your response must be formatted as either "true" or "false".  
    Do not include any additional text or characters in your response.
  """

prompt = PromptTemplate(
    input_variables=["question"],
    template=TEMPLATE
)
llm_chain = prompt | model

input_csv_path = 'boolq_test.csv'  # 替换为实际文件路径
df = pd.read_csv(input_csv_path)


y_true = []  # 真实标签
y_pred = []  # 模型预测标签
score = 0
mismatches = []  # 存储不匹配情况
total = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    question = row['question']
    correct_answer = str(row['answer']).lower()

    result = llm_chain.invoke({
        "question": question
    }).strip()

    result = result.strip().replace('"', '').replace("'", "").lower()  # 去除引号
    if result not in {"true", "false"}:
        result = "I don't know."

    y_pred.append(result)
    y_true.append(correct_answer)

    if result == correct_answer:
        score += 1
    else:
        mismatches.append((correct_answer, result))

    total += 1
    print(f"Total: {total}, Score: {score}, True: {correct_answer}, Predicted: {result}")

# 计算评价指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=np.nan)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=np.nan)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=np.nan)
conf_mat = confusion_matrix(y_true, y_pred)

# 输出结果
print("\nEvaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_mat)

# 打印不匹配的情况
print("\nMismatched Cases:")
for mismatch in mismatches:
    print(f"Expected: {mismatch[0]}, Predicted: {mismatch[1]}")
