import streamlit as st
from rag import answer

st.set_page_config(page_title="PharmaDoc", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .header { padding: 1.5rem 0 0.5rem 0; border-bottom: 2px solid #1a3c5e; margin-bottom: 1.5rem; }
    .header h1 { color: #1a3c5e; font-size: 1.6rem; font-weight: 700; margin: 0; }
    .header p { color: #555; font-size: 0.85rem; margin: 4px 0 0 0; }
    .user-bubble {
        background: #1a3c5e;
        color: #ffffff;
        border-radius: 16px 16px 4px 16px;
        padding: 10px 16px;
        margin: 8px 0 8px 20%;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .bot-bubble {
        background: #ffffff;
        color: #1a1a1a;
        border: 1px solid #dde3ea;
        border-radius: 16px 16px 16px 4px;
        padding: 10px 16px;
        margin: 8px 20% 8px 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .source-line {
        font-size: 0.75rem;
        color: #1a3c5e;
        margin: 2px 0 8px 4px;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ccd3db;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>PharmaDoc</h1>
    <p>Clinical document assistant for pharmacists</p>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "memory" not in st.session_state:
    st.session_state.memory = []

with st.sidebar:
    st.markdown("**PharmaDoc**")
    st.markdown("Model: Llama 3.2")
    st.markdown("Memory: last 3 exchanges")
    st.markdown("Languages: French / English")
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.memory  = []
        st.rerun()
    st.divider()
    st.caption("Place pharmaceutical PDFs in the docs/ folder and run ingest.py to index them.")

for role, text, sources in st.session_state.history:
    if role == "user":
        st.markdown(f'<div class="user-bubble">{text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{text}</div>', unsafe_allow_html=True)
        if sources:
            src_text = " &nbsp;|&nbsp; ".join(sources)
            st.markdown(f'<div class="source-line">Sources: {src_text}</div>', unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input(
            "question",
            placeholder="Ex: Quelles sont les contre-indications de la metformine ?",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    question = user_input.strip()
    st.session_state.history.append(("user", question, []))
    with st.spinner("Searching documents ..."):
        result = answer(question, st.session_state.memory)
    st.session_state.history.append(("assistant", result["answer"], result["sources"]))
    st.session_state.memory.append({"question": question, "answer": result["answer"]})
    st.rerun()