# Importing the libraries
import streamlit as st
import os
import requests
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from huggingface_hub import InferenceClient

# CONFIGURATION
REPO_OWNER = "Abhijeet-Singh-2001"
REPO_NAME = "my-portfolio"
DATA_FOLDER = "data"
HF_TOKEN = os.environ.get("HF_TOKEN")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# FIX: A list of models to try if one fails
MODEL_LIST = [
    "HuggingFaceH4/zephyr-7b-beta",          
    "mistralai/Mistral-7B-Instruct-v0.2",    
    "Qwen/Qwen2.5-7B-Instruct"               
]

st.set_page_config(page_title="Abhijeet's AI", page_icon="")
st.title(" Chat with Abhijeet's AI")

# 1. ROBUST AI CLIENT (With Fallback)
def query_llm_with_fallback(messages):
    client = InferenceClient(token=HF_TOKEN)
    errors = []

    for model in MODEL_LIST:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            # If model is loading, wait and retry once
            if "503" in str(e):
                time.sleep(2)
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=512
                    )
                    return response.choices[0].message.content
                except:
                    pass
            
            errors.append(f"{model}: {str(e)}")
            continue # Try next model

    # If all fail
    return f" All models failed. Details: {'; '.join(errors)}"

# 2. FETCH DATA
@st.cache_resource(ttl=3600)
def fetch_github_content():
    base_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{DATA_FOLDER}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    
    try:
        response = requests.get(base_url, headers=headers)
        if response.status_code != 200:
            st.error(f" GitHub Error: {response.status_code}")
            return []
        
        files = response.json()
        documents = []
        for file in files:
            if isinstance(file, dict) and file['name'].endswith((".md", ".txt")):
                raw_url = file['download_url']
                content_resp = requests.get(raw_url)
                if content_resp.status_code == 200:
                    doc = Document(page_content=content_resp.text, metadata={"source": file['name']})
                    documents.append(doc)
        return documents
    except Exception as e:
        st.error(f" Connection Error: {e}")
        return []

# 3. SETUP DATABASE
@st.cache_resource
def setup_vector_db(_documents):
    if not _documents:
        return None
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(_documents)
    
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# 4. MAIN APP
if not HF_TOKEN:
    st.error(" HF_TOKEN is missing. Please add it in Settings.")
    st.stop()

with st.spinner("Loading knowledge base..."):
    docs = fetch_github_content()
    vector_db = setup_vector_db(docs)

if vector_db:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi There! Hope you're going well. I'm here to give all the detailed information about Abhijeet Singh (Data Scientist)"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Retrieve Context
                search_results = vector_db.similarity_search(prompt, k=3)
                context_text = "\n\n".join([doc.page_content for doc in search_results])
                
                # 2. Prepare Messages
                messages = [
                    {"role": "user", "content": f"Use this context to answer the question:\n\nContext: {context_text}\n\nQuestion: {prompt}"}
                ]

                # 3. Generate Answers
                answer = query_llm_with_fallback(messages)
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.stop()
