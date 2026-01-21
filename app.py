import streamlit as st
import requests
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import os

# CONFIGURATION
REPO_OWNER = "Abhijeet-Singh-2001"
REPO_NAME = "my-portfolio"  
DATA_FOLDER = "data"            
HF_TOKEN = os.environ.get("HF_TOKEN") 

# FUNCTION TO FETCH DATA FROM GITHUB
@st.cache_resource(ttl=3600) 
def fetch_github_content():
    base_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{DATA_FOLDER}"
    response = requests.get(base_url)
    
    if response.status_code != 200:
        st.error(f"Failed to fetch data from GitHub. Status: {response.status_code}")
        return []

    files = response.json()
    documents = []

    for file in files:
        if file['name'].endswith(".md") or file['name'].endswith(".txt"):
            # Get the raw content of the file
            raw_url = file['download_url']
            content_response = requests.get(raw_url)
            if content_response.status_code == 200:
                text = content_response.text
                # Create a LangChain Document
                doc = Document(page_content=text, metadata={"source": file['name']})
                documents.append(doc)
    
    return documents

# SETUP RAG PIPELINE
@st.cache_resource
def setup_rag_chain(_documents):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(_documents)

    # Create Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Vector Store (FAISS)
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Setup LLM (Mistral-7B)
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        huggingfacehub_api_token=HF_TOKEN
    )

    # Create Retrieval Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3})
    )
    return qa

# STREAMLIT UI
st.set_page_config(page_title="Chat with Abhijeet's AI", page_icon="Abhijeet Singh")

st.title("🤖 Chat with Abhijeet's AI Agent")
st.markdown("""
Welcome! I am an AI agent powered by **RAG (Retrieval Augmented Generation)**. 
I have read Abhijeet's GitHub repository and can answer questions about his:
- 📄 **Resume & Experience**
- 🛠️ **Projects (GenAI, NLP, Data Science)**
- 🎓 **Research & Dissertation**
""")

# Load Data
with st.spinner("Fetching latest data from GitHub..."):
    docs = fetch_github_content()

if docs:
    qa_chain = setup_rag_chain(docs)
    
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask me about Abhijeet's skills..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("No data found in the GitHub 'data' folder.")