import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import pandas as pd
from tqdm import tqdm

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

    return scored_queries


input_csv_path = 'hotpotqa_test_qr.csv'  # 替换为实际文件路径
df = pd.read_csv(input_csv_path)

y_true = []  # 真实标签
y_pred = []  # 模型预测标签
mismatches = []  # 存储不匹配情况
total = len(df)  # 总问题数

for index, row in tqdm(df.iterrows(), total=len(df)):
    question = row['question']
    re_queries = row['Rewritten Queries']

    if isinstance(re_queries, str):
        re_queries = eval(re_queries)  # 如果是字符串格式的 list，将其转换为真实的 list

    scored_queries = retrieve_knowledge(question, re_queries, ranker, tokenizer, retriever)



