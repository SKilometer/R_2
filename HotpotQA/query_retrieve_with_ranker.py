import torch
from torch import nn
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from transformers import BertTokenizer, BertModel
from langchain_core.output_parsers import StrOutputParser


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # 1024维度
ollama_llm = 'qwen2.5:32b'
model = Ollama(
    model=ollama_llm,
    temperature=0.1,
)
chromadb = Chroma(persist_directory="./chromadb/chroma_boolq", embedding_function=embeddings)
retriever = chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 5})


class PointwiseRanker(nn.Module):
    def __init__(self, model):
        super(PointwiseRanker, self).__init__()
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

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask).pooler_output
        score = self.fc(outputs)
        return score


model_path = './bert-base-uncased'
tokenizer_path = './bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertModel.from_pretrained(model_path).to(device)
model.eval()

ranker = PointwiseRanker(model).to(device)
ranker = torch.nn.DataParallel(ranker)
ranker_model_path = "./pointwise_ranker/pointwise_ranker.pth"
checkpoint = torch.load(ranker_model_path, map_location=device, weights_only=True)
ranker.module.load_state_dict(checkpoint['model_state_dict'])
print("Model loaded successfully.")

ranker.eval()


# 对重写问题利用ranker进行打分
def score_queries(queries, original_question, context, ranker, tokenizer):
    with torch.no_grad():
        max_length = 512
        scores = []
        for query in queries:
            inputs = tokenizer(
                text=query,
                text_pair=original_question + " // " + context,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = inputs['input_ids'].to(device).squeeze(0)
            attention_mask = inputs['attention_mask'].to(device).squeeze(0)

            score = ranker(input_ids, attention_mask)
            scores.append(score.item())

    return scores


# 根据原始问题进行检索知识，问题重写之后利用ranker进行打分排序，然后返回top-k的重写问题进行向量库匹配
def retrieve_knowledge(question, re_queries, context, ranker, tokenizer):
    """
       插入排序器 ,使用 Ranker 为重写的问题打分
       排序之后返回Top-k的重写问题保存在 reranked_queries中
    """
    scores = score_queries(re_queries, question, context, ranker, tokenizer)
    scored_queries = list(zip(scores, re_queries))
    scored_queries.sort(reverse=True, key=lambda x: x[0])  # 按分数从高到低排序
    print(scored_queries)
    # 选择评分最高的3个问题，如果问题少于3个则选择所有
    top_k_queries = [query for _, query in scored_queries[:min(3, len(scored_queries))]]
    print(top_k_queries)

    retrieved_knowledge = []
    unique_contents = set()  # 用于跟踪已经添加的文档内容
    for requery in top_k_queries:
        retrieved_docs = retriever.invoke(requery)
        for doc in retrieved_docs:
            if doc.page_content not in unique_contents:
                retrieved_knowledge.append(doc.page_content)
                unique_contents.add(doc.page_content)

    return retrieved_knowledge


question = "Is an ocelot a good present for a kindergartener?"
re_queries = ['What is the behavior and social nature of an ocelot?',
        'Is #1 suitable for kindergartners considering their age?']

context1 = retriever.invoke(question)
context2 = retrieve_knowledge(question, re_queries, context1, ranker, tokenizer)


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
rag_chain = rag_qr_prompt | model | StrOutputParser()
rag_qr_chain = rag_qr_prompt | model | StrOutputParser()

result1 = rag_chain.invoke({"question": question, "context": context1}).strip()
print("result1:", result1)
result2 = rag_qr_chain.invoke({"question": question, "context": context2}).strip()

print("result2:", result2)
