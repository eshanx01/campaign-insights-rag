import streamlit as st
from rag_chain import rag_chain

st.set_page_config(page_title="Campaign Insights Assistant", page_icon="📊")
st.title("📊 Campaign Insights RAG Assistant")

# --- Password gate: blocks strangers from spending your OpenAI credit ---
try:
    correct_password = st.secrets["APP_PASSWORD"]
except Exception:
    correct_password = None  # no password set (e.g. running locally)

if correct_password:
    entered = st.text_input("Password", type="password")
    if entered != correct_password:
        if entered:
            st.error("Wrong password.")
        st.stop()
# -----------------------------------------------------------------------

st.write("Ask a question about campaign performance, grounded in campaign reports.")
question = st.text_input("Your question:")

if question:
    with st.spinner("Retrieving and generating answer..."):
        answer = rag_chain.invoke(question)
    st.write("### Answer")
    st.write(answer)
