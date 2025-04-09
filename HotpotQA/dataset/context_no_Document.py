import csv
import os
from tqdm import tqdm
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# 初始化相关组件
embeddings = OllamaEmbeddings(model="nomic-embed-text")
model = Ollama(model='qwen2.5:32b')
chromadb = Chroma(persist_directory="../chromadb/chroma_hotpotqa", embedding_function=embeddings)
retriever = chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# 修改后的处理CSV文件函数
def process_csv(input_csv, output_csv):
    directory = os.path.dirname(output_csv)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(input_csv, 'r', encoding='utf-8-sig') as csv_in, open(output_csv, 'w', newline='', encoding='utf-8-sig') as csv_out:
        reader = csv.DictReader(csv_in)
        writer = csv.DictWriter(csv_out, fieldnames=reader.fieldnames)  # 使用相同的字段名
        writer.writeheader()

        for row in tqdm(reader):
            question = row['Original Question'].strip()

            # 替换Context列
            retrieved_docs = retriever.invoke(question)
            context = [doc.page_content for doc in retrieved_docs]  # 收集所有检索到的文档内容
            context_str = "; ".join(context)  # 将所有内容合并为一个字符串，用分号分隔

            # 更新行数据
            row['Context'] = context_str
            writer.writerow(row)


# 文件路径配置
input_filename = 'question_context_qr_score.csv'
output_filename = 'question_context_qr_score_1.csv'
process_csv(input_filename, output_filename)
