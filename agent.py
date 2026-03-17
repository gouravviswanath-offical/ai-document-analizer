import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def get_qa_chain(db_path="engineering_db"):
    if not os.path.exists(db_path):
        return None
    
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    retriever = db.as_retriever()
    llm = ChatOpenAI()

    template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Context: {context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain

def ask_ai(question, db_path="engineering_db"):
    qa = get_qa_chain(db_path)
    if qa is None:
        return "Vector database not found. Please upload and process a document first."
    
    response = qa.invoke(question)
    return response