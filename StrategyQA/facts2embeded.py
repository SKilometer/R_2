import os
import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=20)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

input_csv_path = './dataset/strategyqa.csv'  # 输入 CSV 文件路径
chroma_persist_dir = './chromadb/chroma_strategyqa'  # Chroma 持久化目录
output_chunks_csv = './dataset/strategyqa_facts_chunks.csv'

if not os.path.exists(chroma_persist_dir):
    os.makedirs(chroma_persist_dir)


def process_exp_column(csv_path):
    print("正在读取 CSV 文件...")
    df = pd.read_csv(csv_path)
    if 'facts' not in df.columns:
        raise ValueError("输入的 CSV 文件中不存在 'facts' 列！")

    documents = []
    chunk_data = []
    print("正在分割 'facts' 列中的文本...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        clean_text = row['facts'].replace('[', '').replace(']', '').replace('"', "'")
        clean_text = clean_text.replace("'s ", "XYZXYZ")
        sentences = clean_text.split("'")
        for sentence in sentences:
            sentence = sentence.replace("XYZXYZ", "'s ")
            clean_sentence = sentence.strip()
            if clean_sentence and not clean_sentence == ",":
                split_chunks = TEXT_SPLITTER.split_text(clean_sentence)
                for chunk in split_chunks:
                    documents.append(chunk)
                    chunk_data.append({"original_index": index, "chunk_content": chunk})

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
print("Chroma 向量库创建成功！")

