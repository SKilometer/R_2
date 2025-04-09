from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

embeddings = OllamaEmbeddings(model="nomic-embed-text")  # 1536维度
ollama_llm = 'qwen2.5:1.5b'
model = Ollama(model=ollama_llm)
ollama_eval_llm = Ollama(model='qwen2.5:32b')
chromadb = Chroma(persist_directory="../chromadb/chroma_boolq", embedding_function=embeddings)
retriever = chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def evaluate_relevance(question, context):
    EVALUATE_TEMPLATE = """
        You are an AI assistant responsible for evaluating whether the provided context is relevant to the given question.  
        Carefully review the following question and its associated context:  

        [Question]: {question}  
        --------------------------  
        [Context]: {context}  

        Your task is to determine:  
            - If the context directly answers the question or includes key information or data that can reasonably help answer it, respond with "relevant".  
            - If the context is unrelated to the question, does not provide sufficient information to answer it, or lacks key details, respond with "irrelevant".  
        Respond using only the word "relevant" or "irrelevant". Do not include any additional text or characters. This ensures clarity and precision.
    """
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=EVALUATE_TEMPLATE
    )
    eval_chain = prompt | ollama_eval_llm
    result = eval_chain.invoke({"question": question, "context": context}).strip().lower()
    if result == "relevant":
        return 1
    elif result == "irrelevant":
        return 0
    else:
        return 1  # 默认返回0以确保返回值为整数


def retrieve_relevant_knowledge(question, retriever):
    docs = retriever.invoke(question)
    relevant_docs = []
    for doc in docs:
        relevance = evaluate_relevance(question, doc)
        if relevance == 1:
            relevant_docs.append(doc.page_content)

    return relevant_docs if relevant_docs else None


RAG_TEMPLATE = """
        Based on the provided question and context, determine whether the following question is true or false:  
        Your response must be formatted as either "true" or "false".  
        Do not include any additional text or characters in your response.

        **Context**: {context}  
        -----------------  
        **Question**: {question}  

        Based on the provided question and context, determine whether the following question is true or false:  
        Your response must be formatted as either "true" or "false".  
        Do not include any additional text or characters in your response.
      """
rag_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=RAG_TEMPLATE
)
rag_eavl_chain = rag_prompt | model | StrOutputParser()

rag_chain = rag_prompt | model | StrOutputParser()

input_csv_path = 'boolq_test.csv'
df = pd.read_csv(input_csv_path)

y_true = []  # 真实标签
y_pred = []  # 模型预测标签
score = 0
mismatches = []  # 存储不匹配情况
total = 0

count = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    question = row['question']
    correct_answer = str(row['answer']).lower()

    context = retriever.invoke(question)
    relevant_context = retrieve_relevant_knowledge(question, retriever)

    if relevant_context is not None:
        result = rag_eavl_chain.invoke({
            "question": question,
            "context": relevant_context
        }).strip()
    else:
        count += 1
        result = rag_chain.invoke({
            "question": question,
            "context": context
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
print("Empety relevant_context count", count)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_mat)

# 打印不匹配的情况
print("Mismatched Cases:")
for mismatch in mismatches:
    print(f"Expected: {mismatch[0]}, Predicted: {mismatch[1]}")
