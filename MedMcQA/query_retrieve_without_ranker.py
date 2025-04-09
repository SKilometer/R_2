from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

embeddings = OllamaEmbeddings(model="nomic-embed-text")
ollama_llm = 'qwen2.5:7b'
model = Ollama(
    model=ollama_llm,
    temperature=0.1,
)

chromadb = Chroma(persist_directory="./chromadb/chroma_medmcqa", embedding_function=embeddings)
retriever = chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 10})


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
        Based on the provided question and context, select the single correct answer from the following options:  
        [0: Antibiotics, 1:Observation, 2: Sting operation, 3: Ureteric reimplantation]  
        Your response must strictly be one of the following: "0", "1", "2", or "3".  
        Provide only the option number as your response, without any additional text or characters.  

        **Context**: {context}  
        -----------------  
        **Question**: {question}  

        Based on the provided question and context, select the single correct answer from the following options:  
        [0: Antibiotics, 1:Observation, 2: Sting operation, 3: Ureteric reimplantation]  
        Your response must strictly be one of the following: "0", "1", "2", or "3".  
        Provide only the option number as your response, without any additional text or characters.  
      """
rag_qr_prompt = PromptTemplate(
    input_variables=["question", "opa", "opb", "opc", "opd", "context"],
    template=TEMPLATE
)
rag_chain = rag_qr_prompt | model | StrOutputParser()

rag_qr_chain = rag_qr_prompt | model | StrOutputParser()

# cancer
print("before re\n ")
question = "A 60-year diabetic & hypertensive male with second-grade prostatism admitted for prostatectomy developed myocardial infarction. Treatment now would be -"
opa = "Finasteride"
opb = "Terazocin"
opc = "Finasteride and terazocin"
opd = "Diethyl stilbestrol"
re_queries = [
    '1. What would be the most appropriate management strategy for a 60-year-old male patient with a history of diabetes and hypertension, who has been diagnosed with second-grade prostatism and is scheduled for prostatectomy, but has subsequently developed a myocardial infarction, taking into account the need to balance the treatment of his cardiac condition with the planned surgical intervention?',
    '2. In the context of a 60-year-old diabetic and hypertensive male patient with second-grade prostatism who is awaiting prostatectomy and has recently experienced a myocardial infarction, what pharmacological and non-pharmacological interventions would be recommended to stabilize his cardiovascular status while also considering the implications for his upcoming surgical procedure?',
    '3. How would the development of a myocardial infarction in a 60-year-old male patient with diabetes, hypertension, and second-grade prostatism, who is an inpatient awaiting prostatectomy, influence the choice of perioperative care, including the potential need for adjustments to his antidiabetic, antihypertensive, and antiplatelet therapies?',
    '4. Considering the complexities of managing a patient with multiple comorbidities, what would be the optimal approach to treating a myocardial infarction in a 60-year-old diabetic and hypertensive male with second-grade prostatism who is scheduled for prostatectomy, focusing on minimizing risks associated with both the cardiac event and the surgical procedure?',
    '5. For a 60-year-old male patient with a history of diabetes and hypertension, presenting with second-grade prostatism and admitted for elective prostatectomy, who then develops an acute myocardial infarction, what would be the recommended sequence and timing of interventions to address his cardiac condition, and how would these interventions impact the planning and execution of his prostatectomy, ensuring the best possible outcomes for both conditions?']

print("question:", question)
context1 = retriever.invoke(question)   # 原
print(context1)
result1 = rag_chain.invoke({"question": question,
                            "opa": opa,
                            "opb": opb,
                            "opc": opc,
                            "opd": opd,
                            "context": context1}).strip()
print("result1: " + result1)
print("\n after re\n ")
context2 = retrieve_knowledge(re_queries, retriever)    # qr
print(context2)
result2 = rag_qr_chain.invoke({"question": question,
                               "opa": opa,
                               "opb": opb,
                               "opc": opc,
                               "opd": opd,
                               "context": context2}).strip()
print("result2: " + result2)
