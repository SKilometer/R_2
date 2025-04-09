import ast
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import numpy as np

embeddings = OllamaEmbeddings(model="nomic-embed-text")
ollama_llm = 'qwen2.5:7b'
model = Ollama(
    model=ollama_llm,
    temperature=0.1,
)

chromadb = Chroma(persist_directory="../chromadb/chroma_medmcqa", embedding_function=embeddings)
retriever = chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def retrieve_knowledge(re_queries, retriever):
    re_queries.sort(reverse=True, key=lambda x: x[1])
    top_k_queries = [query for query, _ in re_queries[:min(3, len(re_queries))]]
    retrieved_knowledge = []
    unique_contents = set()  # 用于跟踪已经添加的文档内容

    for requery in top_k_queries:
        retrieved_docs = retriever.invoke(requery)
        for doc in retrieved_docs:
            if doc.page_content not in unique_contents:
                retrieved_knowledge.append(doc.page_content)
                unique_contents.add(doc.page_content)

    return retrieved_knowledge


TEMPLATE = """
        Based on the provided question and context, select the single correct answer from the following options:  
        [0: {opa}, 1: {opb}, 2: {opc}, 3: {opd}]  
        Your response must strictly be one of the following: "0", "1", "2", or "3".  
        Provide only the option number as your response, without any additional text or characters.  

        **Context**: {context}  
        -----------------  
        **Question**: {question}  

        Based on the provided question and context, select the single correct answer from the following options:  
        [0: {opa}, 1: {opb}, 2: {opc}, 3: {opd}]  
        Your response must strictly be one of the following: "0", "1", "2", or "3".  
        Provide only the option number as your response, without any additional text or characters.  
      """
rag_qr_prompt = PromptTemplate(
    input_variables=["question", "opa", "opb", "opc", "opd", "context"],
    template=TEMPLATE
)
rag_qr_chain = rag_qr_prompt | model | StrOutputParser()

input_csv_path = 'medmcqa_test_qr_score.csv'
df = pd.read_csv(input_csv_path)

y_true = []  # 真实标签
y_pred = []  # 模型预测标签
score = 0
mismatches = []  # 存储不匹配情况
total = 0


for _, row in tqdm(df.iterrows(), total=len(df)):
    question = row['question']
    opa = row['opa']
    opb = row['opb']
    opc = row['opc']
    opd = row['opd']
    re_queries_scores = ast.literal_eval(row['Rewritten Queries with Scores'])
    correct_answer = str(row['cop'])

    context = retrieve_knowledge(re_queries_scores, retriever)

    result = rag_qr_chain.invoke({
        "question": question,
        "opa": opa,
        "opb": opb,
        "opc": opc,
        "opd": opd,
        "context": context
    }).strip()

    result = result.strip().replace('"', '').replace("'", "")  # 去除引号
    if result not in {"0", "1", "2", "3"}:
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
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_mat)

# 打印不匹配的情况
print("Mismatched Cases:")
for mismatch in mismatches:
    print(f"Expected: {mismatch[0]}, Predicted: {mismatch[1]}")

