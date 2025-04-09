import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
ollama_llm = 'qwen2.5:7b'
model = Ollama(
    model=ollama_llm,
    temperature=0.1,
)

chromadb = Chroma(persist_directory="./chromadb/chroma_medmcqa", embedding_function=embeddings)
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
        outputs = self.bert(input_ids, attention_mask=attention_mask).pooler_output
        score = self.fc(outputs)
        return score


model_path = './Bio_ClinicalBERT'
tokenizer_path = './Bio_ClinicalBERT'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModel.from_pretrained(model_path).to(device)
model.eval()

ranker = PointwiseRanker(model).to(device)
ranker = torch.nn.DataParallel(ranker)
ranker_model_path = "./pointwise_ranker/pointwise_ranker_2.pth"
checkpoint = torch.load(ranker_model_path, map_location=device)
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
    scored_queries.sort(reverse=True, key=lambda x: x[0])   # 按分数从高到低排序
    print(scored_queries)
    # 选择评分最高的3个问题，如果问题少于3个则选择所有
    top_k_queries = [query for _, query in scored_queries[:min(3, len(scored_queries))]]
    print(top_k_queries)

    # 检索排序后的重写问题context
    retrieved_knowledge = []
    unique_contents = set()  # 用于跟踪已经添加的文档内容
    for requery in top_k_queries:
        retrieved_docs = retriever.invoke(requery)
        for doc in retrieved_docs:
            if doc.page_content not in unique_contents:
                retrieved_knowledge.append(doc.page_content)
                unique_contents.add(doc.page_content)

    return retrieved_knowledge


question = "A 60-year diabetic & hypertensive male with second-grade prostatism admitted for prostatectomy developed myocardial infarction. Treatment now would be -"

re_queries = [
    '1. What would be the most appropriate management strategy for a 60-year-old male patient with a history of diabetes and hypertension, who has been diagnosed with second-grade prostatism and is scheduled for prostatectomy, but has subsequently developed a myocardial infarction, taking into account the need to balance the treatment of his cardiac condition with the planned surgical intervention?',
    '2. In the context of a 60-year-old diabetic and hypertensive male patient with second-grade prostatism who is awaiting prostatectomy and has recently experienced a myocardial infarction, what pharmacological and non-pharmacological interventions would be recommended to stabilize his cardiovascular status while also considering the implications for his upcoming surgical procedure?',
    '3. How would the development of a myocardial infarction in a 60-year-old male patient with diabetes, hypertension, and second-grade prostatism, who is an inpatient awaiting prostatectomy, influence the choice of perioperative care, including the potential need for adjustments to his antidiabetic, antihypertensive, and antiplatelet therapies?',
    '4. Considering the complexities of managing a patient with multiple comorbidities, what would be the optimal approach to treating a myocardial infarction in a 60-year-old diabetic and hypertensive male with second-grade prostatism who is scheduled for prostatectomy, focusing on minimizing risks associated with both the cardiac event and the surgical procedure?',
    '5. For a 60-year-old male patient with a history of diabetes and hypertension, presenting with second-grade prostatism and admitted for elective prostatectomy, who then develops an acute myocardial infarction, what would be the recommended sequence and timing of interventions to address his cardiac condition, and how would these interventions impact the planning and execution of his prostatectomy, ensuring the best possible outcomes for both conditions?']

opa = "Finasteride"
opb = "Terazocin"
opc = "Finasteride and terazocin"
opd = "Diethyl stilbestrol"

print("question:" + question)

context1 = retriever.invoke(question)

context2 = retrieve_knowledge(question, re_queries, context1, ranker, tokenizer)


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
rag_chain = rag_qr_prompt | model | StrOutputParser()
rag_qr_chain = rag_qr_prompt | model | StrOutputParser()

result1 = rag_chain.invoke({"question": question,
                              "opa": opa,
                              "opb": opb,
                              "opc": opc,
                              "opd": opd,
                              "context": context1}).strip()
print("result1:", result1)
result2 = rag_qr_chain.invoke({"question": question,
                              "opa": opa,
                              "opb": opb,
                              "opc": opc,
                              "opd": opd,
                              "context": context2}).strip()

print("result2:", result2)
