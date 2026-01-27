# my-portfolio
### Chat with Abhijeet's AI
### model - (BAAI/bge-small-en-v1.5)

This is an AI-powered portfolio chatbot that answers questions about my professional experience, skills, and background using a **RAG (Retrieval Augmented Generation)** pipeline.
Try it live here: https://huggingface.co/spaces/Abhijeet2/abhijeet-portfolio

## How It Works
This project connects my static portfolio data (markdown files) with a Large Language Model (LLM) to create an interactive chat experience.

## The Architecture
1.  Data Ingestion: The app fetches text files (".md", ".txt") directly from this GitHub repository.
2.  Chunking: The text is split into smaller, meaningful chunks using "RecursiveCharacterTextSplitter".
3.  Embedding: These chunks are converted into numerical vectors using "FastEmbed" (BAAI/bge-small-en-v1.5).
4.  Vector Database: The vectors are stored in a temporary "FAISS" database for fast searching.
5.  Retrieval: When you ask a question, the app finds the most relevant text chunks.
6.  Generation: The relevant text + your question are sent to the **bge-small-en-v1.5** LLM via the Hugging Face Inference API to generate a human-like answer.

## Tech Stack
* Frontend: (https://streamlit.io/) (for the chat interface) 
* LLM: Hugging Face "InferenceClient" (Serverless API)
* Embeddings: "fastembed" (Local, lightweight, and fast)
* Vector DB: "FAISS" (Facebook AI Similarity Search)
* Framework: "LangChain" (for document processing)

## How to Run Locally

1.  Clone the repository:
    "bash
    git clone [https://github.com/Abhijeet-Singh-2001/my-portfolio.git](https://github.com/Abhijeet-Singh-2001/my-portfolio.git)
    cd my-portfolio
    "

2.  Install dependencies:
    "bash
    pip install -r requirements.txt
    "

3.  Set up keys:
    Create a ".env" file and add your tokens:
    "
    HF_TOKEN=your_huggingface_token_here
    GITHUB_TOKEN=your_github_token_here
    "

4.  Run the app:
    "bash
    streamlit run app.py
    "

  Local URL: http://localhost:8501
  Network URL: http://10.108.119.112:8501
  External URL: http://44.209.54.138:8501
  
## Project Structure

* "data/": Contains the markdown files with my portfolio info.
* "app.py": The main Python application logic.
* "requirements.txt": List of Python libraries used.

---
# * Created by Abhijeet_Singh *

Demo
<img width="1919" height="914" alt="image" src="https://github.com/user-attachments/assets/0c984f32-a8b2-4273-8cdd-8a52290754a0" />
