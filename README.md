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
