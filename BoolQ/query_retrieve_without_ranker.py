from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

embeddings = OllamaEmbeddings(model="nomic-embed-text")  # 1024维度
ollama_llm = 'qwen2.5:7b'
model = Ollama(
    model=ollama_llm,
    temperature=0.1,
)
chromadb = Chroma(persist_directory="./chromadb/chroma_boolq", embedding_function=embeddings)
retriever = chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# 根据原始问题进行检索知识
def retrieve_knowledge(re_queries, retriever):
    retrieved_knowledge = []
    unique_contents = set()  # 用于跟踪已经添加的文档内容

    for requery in re_queries:
        retrieved_docs = retriever.invoke(requery)
        for doc in retrieved_docs:
            if doc.page_content not in unique_contents:
                retrieved_knowledge.append(doc.page_content)
                unique_contents.add(doc.page_content)

    return retrieved_knowledge


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

question = "Did Stone Cold Steve Austin wrestle in three different centuries?".strip()
re_queries = ['When did Stone Cold Steve Austin start wrestling?',
    'When did Stone Cold Steve Austin stop wrestling?',
    'In what century is #1?' 'In what century is #2?',
    'Is #4 minus #3 greater than 1?']


print("before re\n ")
context1 = retriever.invoke(question)
print(context1)
result1 = rag_chain.invoke({"question": question, "context": context1}).strip()
print("result1: " + result1)
print("\n after re\n ")
context2 = retrieve_knowledge(re_queries, retriever)    # qr
print(context2)
result2 = rag_qr_chain.invoke({"question": question, "context": context2}).strip()
print("result2: " + result2)

