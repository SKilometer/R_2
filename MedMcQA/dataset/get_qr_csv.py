import uuid
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler, CallbackManagerForRetrieverRun
from langchain_core.prompts import PromptTemplate
import pandas as pd
from tqdm import tqdm

embeddings = OllamaEmbeddings(model="nomic-embed-text")
ollama_llm = 'qwen2.5:1.5b'
model = Ollama(model=ollama_llm, temperature=0.1)
run_manager = CallbackManagerForRetrieverRun(
    run_id=uuid.uuid4(),
    handlers=[StreamingStdOutCallbackHandler()],
    inheritable_handlers=[]
)

chromadb = Chroma(persist_directory="../chromadb/chroma_medmcqa", embedding_function=embeddings)
retriever = chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def re_query(question):
    multi_template = """As an AI assistant specializing in the medical field, my role is to analyze and reframe medical questions to enhance their clarity, precision, and applicability. 
            I will approach each question step-by-step, breaking it down or rewriting it from multiple perspectives while maintaining accuracy and professionalism. 
            All rewritten questions will adhere to medical terminology and standards, ensuring they are rigorous and practical. 
            Each rewritten question will be sequentially numbered, fully articulated, and logically structured. 
            A total of five distinct rewritten questions will be generated, each offering a unique perspective or phrasing.
            Questions will be separated by line breaks to ensure readability. 
            Below is the original question to be rewritten: {question}
            """

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=multi_template,
    )

    retriever_llm = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=model,
        prompt=QUERY_PROMPT
    )

    rewritten_queries = retriever_llm.generate_queries(question, run_manager)  # 原始问题重写生成rewritten_queries
    rewritten_queries = [query.strip() for query in rewritten_queries if query.strip()]  # 去除空行

    return rewritten_queries


input_csv_path = 'medmcqa_test.csv'
output_csv_path = 'medmcqa_test_qr.csv'

df = pd.read_csv(input_csv_path)
rewritten_queries = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    question = row['question']

    rewritten = re_query(question)
    rewritten_queries.append(rewritten)

df['Rewritten Queries'] = rewritten_queries

df.to_csv(output_csv_path, index=False)
print(f"处理完成，新CSV文件已保存至：{output_csv_path}")
