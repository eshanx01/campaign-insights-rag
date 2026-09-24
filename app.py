import streamlit as st
from rag_chain import rag_chain

st.set_page_config(page_title="Campaign Insights Assistant", page_icon="📊")
st.title("📊 Campaign Insights RAG Assistant")
st.write("Ask a question about campaign performance, grounded in real campaign data.")

question = st.text_input("Your question:")

if question:
    with st.spinner("Retrieving and generating answer..."):
        answer = rag_chain.invoke(question)
    st.write("### Answer")
    st.write(answer)