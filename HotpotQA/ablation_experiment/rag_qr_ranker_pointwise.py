import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import pandas as pd
from tqdm import tqdm
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

embeddings = OllamaEmbeddings(model="nomic-embed-text")  # 1024维度
ollama_llm = 'qwen2.5:32b'
model = Ollama(
    model=ollama_llm,
    temperature=0.1,
)
chromadb = Chroma(persist_directory="../chromadb/chroma_hotpotqa", embedding_function=embeddings)
retriever = chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 5})


class PointwiseRanker(nn.Module):
    def __init__(self, model):
        super(PointwiseRanker, self).__init__()
        self.bert = model
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),  # hidden_size = 768
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        score = self.fc(cls_emb)
        return score


model_path = '../bert-base-uncased'
tokenizer_path = '../bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertModel.from_pretrained(model_path).to(device)
model.eval()

ranker = PointwiseRanker(model).to(device)
ranker = torch.nn.DataParallel(ranker)
ranker_model_path = "../ranker/pointwise_ranker/pointwise_ranker.pth"
checkpoint = torch.load(ranker_model_path, map_location=device, weights_only=True)
ranker.module.load_state_dict(checkpoint['model_state_dict'])
print("pointwise_ranker model loaded successfully.")
ranker.eval()


def ranker_queries(re_queries, question, context, ranker, tokenizer):
    with torch.no_grad():
        max_length = 512
        scored_queries = []
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

            score = ranker(input_ids, attention_mask).squeeze(1)
            scored_queries.append((query, score.item()))
        scored_queries.sort(reverse=True, key=lambda x: x[1])

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
        # Role:
            You are an artificial intelligence Q&A assistant. I will prepare open-domain or multi-hop questions for you, and your role is to help me answer these questions succinctly.

        # Input and Output：
            Based on the provided question and context, , please provide a concise and accurate answer below.
            Ensure that your response is directly relevant, extremely succinct, and encapsulates the answer in the fewest words possible.

            **Context**: {context}  
            -----------------  
            **Question**: {question}  

            Based on the provided question and context, , please provide a concise and accurate answer below.
            Ensure that your response is directly relevant, extremely succinct, and encapsulates the answer in the fewest words possible.


        # Output Example：
            Below are five example questions with standard answers to help you learn the answering technique.

            Question1：South Dade High School is located between what two national parks?
            Answer1：Biscayne National Park to the east and Everglades National Park to the west

            Question2：Who was the king of England who became king after the death of his younger brother and whose reign was chaotic due to rivalry with relatives until his own death a few decades later?
            Answer2：Stephen of Blois

            Question3：Are Ferocactus and Silene both types of plant?
            Answer3：yes

            Question4：What organization does Sandra Pizzarello and Doctorate have in common?
            Answer4：University

            Question5：How many episodes were there of the TV series where Julianna Margulies had the role of Carol Hathaway ?
            Answer5：331 episodes

        # Rules：
        - Use the shortest language possible, avoiding any superfluous words.
        - Ensure that no extraneous information unrelated to the question is included in your answer.
        - Use English words only, avoiding any Chinese words.
        - Use the correct, complete, and appropriate singular or plural forms.
        - Use the correct tense and voice in your answers.

      """
rag_qr_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=TEMPLATE
)
rag_qr_chain = rag_qr_prompt | model | StrOutputParser()

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
    re_queries = row['Rewritten Queries']
    correct_answer = row['answer']
    correct_answer = clean_answer(correct_answer)

    if isinstance(re_queries, str):
        re_queries = eval(re_queries)  # 如果是字符串格式的 list，将其转换为真实的 list

    retrieve_knowledge = retrieve_knowledge(question, re_queries, ranker, tokenizer, retriever)

    result = rag_qr_chain.invoke({
        "question": question,
        "context": retrieve_knowledge
    }).replace('"', '').replace('.', '').replace("'", "")

    result = clean_answer(result)
    if result not in {"true", "false"}:
        result = "I don't know."

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
