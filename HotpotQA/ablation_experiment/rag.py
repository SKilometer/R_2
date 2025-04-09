from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
import pandas as pd
import re

embeddings = OllamaEmbeddings(model="nomic-embed-text")  # 1024维度
ollama_llm = 'qwen2.5:1.5b'

model = Ollama(
    model=ollama_llm,
    temperature=0.1,
)

chromadb = Chroma(persist_directory="../chromadb/chroma_hotpotqa", embedding_function=embeddings)
retriever = chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

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

rag_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=TEMPLATE
)
rag_chain = rag_prompt | model | StrOutputParser()


def format_context(docs):
    docs_cleaned = []
    for doc in docs:
        docs_cleaned.append(doc.page_content)

    return docs_cleaned


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
    correct_answer = row['answer']
    correct_answer = clean_answer(correct_answer)

    retrieved_docs = retriever.invoke(question)
    context = format_context(retrieved_docs)

    result = rag_chain.invoke({"question": question, "context": context}).replace('"', '').replace('.', '').replace("'", "")
    result = clean_answer(result)

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
