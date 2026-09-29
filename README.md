# Campaign Insights RAG Assistant

A Retrieval-Augmented Generation system that answers natural-language questions
about marketing campaign performance, grounded in structured (CSV) and
unstructured (report) campaign data.

## Architecture
Documents → Chunking (RecursiveCharacterTextSplitter) → Embeddings (OpenAI) →
Vector Store (Chroma) → Retrieval (top-k similarity search) → Prompt-constrained
Generation (GPT-4o-mini) → Answer

## Evaluation
Evaluated using RAGAS across faithfulness, answer relevancy, and context
precision. See `ragas_scores.csv` for results.

## Tech Stack
Python, LangChain, Chroma, OpenAI, RAGAS, Streamlit

## Live Demo
https://campaign-insights-rag-ucxmxdgeadfwd7xzurmgnl.streamlit.app/

https://github.com/user-attachments/assets/da9dc825-0c38-4301-8955-22b164009bc0

https://github.com/user-attachments/assets/87479896-19ac-41da-b51d-7ab36d4ddd42









