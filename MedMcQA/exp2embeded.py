import os
import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

input_csv_path = './dataset/medmcqa_filtered.csv'  # 输入 CSV 文件路径
chroma_persist_dir = './chromadb/chroma_medmcqa'  # Chroma 持久化目录
output_chunks_csv = './dataset/exp_chunks.csv'

if not os.path.exists(chroma_persist_dir):
    os.makedirs(chroma_persist_dir)


def process_exp_column(csv_path):
    print("正在读取 CSV 文件...")
    df = pd.read_csv(csv_path)
    if 'exp' not in df.columns:
        raise ValueError("输入的 CSV 文件中不存在 'exp' 列！")

    documents = []
    chunk_data = []
    print("正在分割 'exp' 列中的文本...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = row['exp']
        if pd.isna(text) or not isinstance(text, str):  # 检查文本是否为空或非字符串
            continue
        split_chunks = TEXT_SPLITTER.split_text(text)
        for chunk in split_chunks:
            documents.append(chunk)
            chunk_data.append({"original_index": index, "chunk_content": chunk})
        # documents.extend(split_chunks)

    chunk_df = pd.DataFrame(chunk_data)
    chunk_df.to_csv(output_chunks_csv, index=False, encoding='utf-8-sig')
    print(f"分割后的文本块已保存到：{output_chunks_csv}")
    print(f"分割完成！共处理 {len(documents)} 个文档块。")
    return documents


exp_documents = process_exp_column(input_csv_path)
chroma_db = Chroma.from_texts(
        texts=exp_documents,
        embedding=embeddings,
        persist_directory=chroma_persist_dir
    )
print("Chroma 向量库创建成功！")        # 25714

