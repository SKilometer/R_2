import csv
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

"""
    生成（question，Sorted（query，score））的文件
    读取medmcqa_train_qr.csv文件，依次读取question，
    对于一个原始问题，先进行问题重写，生成n个重写问题
    每个重写问题与向量库进行匹配检索k个相关的chunks
    将（原始问题，chunk）送入evaluate_relevance进行0-1打分
    将k个chunks得分相加得到一个重写问题的得分
    最后将（question，Sorted（query，score））进行保存 
    生成medmcqa_question_qr_score.csv文件
    
"""

embeddings = OllamaEmbeddings(model="nomic-embed-text")
ollama_llm = 'qwen2.5:32b'
model = Ollama(model=ollama_llm)
ollama_eval_llm = Ollama(model=ollama_llm)

chromadb = Chroma(persist_directory="./chromadb/chroma_medmcqa", embedding_function=embeddings)
retriever = chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 10})


# evaluation 使用大模型进行打分输入（question, context）进行0-1打分
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


# 将问题重写，得到重写问题进行评估打分，按照得分高低进行排序并且返回排好序的重写问题列表
def get_rank(question, re_queries, retriever):
    requery_scores = []  # [(q,s),(q,s),(q,s)...]
    for requery in re_queries:
        # 计算每个重写问题的得分
        total_score = 0
        retrieved_docs = retriever.invoke(requery)  # 每个重写问题找到top-k个chunks
        for doc in retrieved_docs:
            score = evaluate_relevance(question, doc)
            total_score += score
        requery_scores.append((requery, total_score))  # [(q,s),(q,s),(q,s)...]

    return requery_scores


# 读取带有原始问题的csv文件，读取question项，进行打分排序，将得到的排序和原始问题一起返回到csv文件
def process_csv(input_csv, output_csv):
    directory = os.path.dirname(output_csv)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(input_csv, 'r', encoding='utf-8-sig') as csv_in, open(output_csv, 'w', newline='', encoding='utf-8-sig') as csv_out:
        reader = csv.DictReader(csv_in)
        writer = csv.writer(csv_out)
        writer.writerow(["Original Question", "Context", "Rewritten Queries with Scores"])
        for row in tqdm(reader):
            question = row.get("question", "").strip()
            re_queries = row.get("Rewritten Queries", "").strip()

            if isinstance(re_queries, str):
                re_queries = eval(re_queries)  # 如果是字符串格式的 list，将其转换为真实的 list

            # print(question)
            if question:
                context = retriever.invoke(question)
                queries_scores = get_rank(question, re_queries, retriever)
                writer.writerow([question, context, queries_scores])


medmaqa_train_filename = './dataset/medmcqa_train_qr.csv'
csv_filename = './dataset/question_context_qr_score.csv'
process_csv(medmaqa_train_filename, csv_filename)

