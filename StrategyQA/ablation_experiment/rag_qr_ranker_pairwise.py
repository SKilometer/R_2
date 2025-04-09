from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

embeddings = OllamaEmbeddings(model="nomic-embed-text")  # 1024维度
ollama_llm = 'qwen2.5:32b'
model = Ollama(
    model=ollama_llm,
    temperature=0.1,
)
chromadb = Chroma(persist_directory="../chromadb/chroma_strategyqa", embedding_function=embeddings)
retriever = chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 5})


class RankNET(nn.Module):
    def __init__(self, model):
        super(RankNET, self).__init__()
        self.bert = model
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),  # hidden_size = 768
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def get_score(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        score = self.fc(cls_emb)
        return score.squeeze(-1)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        score1 = self.get_score(input_ids=input_ids1, attention_mask=attention_mask1)
        score2 = self.get_score(input_ids=input_ids2, attention_mask=attention_mask2)
        diff = score1 - score2
        return diff

    def predict(self, input_ids, attention_mask):
        return self.get_score(input_ids, attention_mask)


model_path = '../bert-base-uncased'
tokenizer_path = '../bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertModel.from_pretrained(model_path).to(device)
model.eval()

ranker = RankNET(model).to(device)
ranker = torch.nn.DataParallel(ranker)
ranker_model_path = "../ranker/pairwise_ranker/pairwise_ranker.pth"
checkpoint = torch.load(ranker_model_path, map_location=device, weights_only=True)
ranker.module.load_state_dict(checkpoint['model_state_dict'])
print("pairwise_ranker model loaded successfully.")
ranker.eval()


def ranker_queries(re_queries, question, context, ranker, tokenizer):
    with torch.no_grad():
        max_length = 512
        scored_queries = []

        # 计算每个重写问题的得分
        for query in re_queries:
            inputs = tokenizer(
                text=query,
                text_pair=question + "[SEP]" + context,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            input_ids = inputs['input_ids'].to(device).squeeze(0)
            attention_mask = inputs['attention_mask'].to(device).squeeze(0)

            score = ranker.predict(input_ids, attention_mask).item()
            scored_queries.append((query, score))

        scored_queries.sort(key=lambda x: x[1], reverse=True)

    return scored_queries


def retrieve_knowledge(question, re_queries, ranker, tokenizer, retriever):
    context = retriever.invoke(question)
    """
        插入排序器 ,使用 Ranker为重写的问题打分
        排序之后返回Top-k的重写问题保存在 reranked_queries中
    """
    scored_queries = ranker_queries(re_queries, question, context, ranker, tokenizer)
    print("scored_queries:", scored_queries)

    # 选择评分最高的3个问题，如果问题少于3个则选择所有
    top_k_queries = [query for query, _ in scored_queries[:min(3, len(scored_queries))]]
    print("Top-k queries:", top_k_queries)

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
rag_qr_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=TEMPLATE
)
rag_qr_chain = rag_qr_prompt | model | StrOutputParser()

input_csv_path = 'strategyqa_qr_test.csv'
df = pd.read_csv(input_csv_path)

y_true = []  # 真实标签
y_pred = []  # 模型预测标签
score = 0
mismatches = []  # 存储不匹配情况
total = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    question = row['question']
    re_queries = row['Rewritten Queries']
    correct_answer = str(row['answer']).lower()

    if isinstance(re_queries, str):
        re_queries = eval(re_queries)  # 如果是字符串格式的 list，将其转换为真实的 list

    retrieve_knowledge = retrieve_knowledge(question, re_queries, ranker, tokenizer, retriever)

    result = rag_qr_chain.invoke({
        "question": question,
        "context": retrieve_knowledge
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
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_mat)

# 打印不匹配的情况
print("Mismatched Cases:")
for mismatch in mismatches:
    print(f"Expected: {mismatch[0]}, Predicted: {mismatch[1]}")
