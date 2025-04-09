import os
import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

input_csv_path = './dataset/hotpotqa.csv'  # 输入 CSV 文件路径
chroma_persist_dir = './chromadb/chroma_hotpotqa'  # Chroma 持久化目录
output_chunks_csv = './dataset/context_chunks.csv'

if not os.path.exists(chroma_persist_dir):
    os.makedirs(chroma_persist_dir)


def process_exp_column(csv_path):
    print("正在读取 CSV 文件...")
    df = pd.read_csv(csv_path)
    if 'context' not in df.columns:
        raise ValueError("输入的 CSV 文件中不存在 'context' 列！")

    documents = []
    chunk_data = []
    print("正在分割 'context' 列中的文本...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = row['context']
        if pd.isna(text) or not isinstance(text, str):  # 检查文本是否为空或非字符串
            continue
        documents_list = text.split('\n')
        documents_list = [doc for doc in documents_list if doc.strip() != '']

        for doc_index, doc in enumerate(documents_list):
            doc = str(doc)
            match = re.search(r'Document \(Title: (.+?)\):', doc)
            if match:
                title = match.group(1).strip()  # 提取标题并去除前后空格

            else:
                title = "--"
            cleaned_text = re.sub(r'Document \(Title: .+?\): ', '', doc)
            split_chunks = TEXT_SPLITTER.split_text(cleaned_text)  # 对每个元素进行递归分割

            for chunk in split_chunks:
                chunk = f"({title}): {chunk}"
                documents.append(chunk)
                chunk_data.append({"original_index": index, "chunk_content": chunk})

    chunk_df = pd.DataFrame(chunk_data)
    chunk_df.to_csv(output_chunks_csv, index=False, encoding='utf-8-sig')
    print(f"分割后的文本块已保存到：{output_chunks_csv}")
    print(f"分割完成！共处理 {len(documents)} 个文档块。")
    return documents


exp_documents = process_exp_column(input_csv_path)      # 7066
chroma_db = Chroma.from_texts(
        texts=exp_documents,
        embedding=embeddings,
        persist_directory=chroma_persist_dir
    )
print("Chroma 向量库创建成功！")

