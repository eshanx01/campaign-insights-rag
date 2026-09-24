import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Load all the .txt campaign reports
loader = DirectoryLoader("reports", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

# Split long documents into smaller overlapping chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=100,
)
chunks = text_splitter.split_documents(documents)

print(f"Loaded {len(documents)} documents, split into {len(chunks)} chunks.")

# Turn chunks into embeddings and store them in a local vector database
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

print("Vector store created and saved to ./chroma_db")