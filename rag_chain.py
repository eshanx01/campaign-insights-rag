import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Reconnect to the vector store you built in ingest.py
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# The prompt template — this is where "prompt engineering" happens
prompt = ChatPromptTemplate.from_template("""
You are a marketing analytics assistant. Answer the question using ONLY
the context provided below. If the context doesn't contain the answer,
say so clearly rather than guessing.

Context:
{context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    question = "Which campaigns underperformed and why?"
    answer = rag_chain.invoke(question)
    print(answer)