import streamlit as st
import os
import io
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
from sentence_transformers import SentenceTransformer

st.title("Q&A with PDF Support")

os.environ['GOOGLE_API_KEY'] = "*****************"  
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

model = genai.GenerativeModel('gemini-pro')

sentence_model = SentenceTransformer('all-mpnet-base-v2')

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about the uploaded PDF document."
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text("text")
    return text

def retrieve_relevant_chunks(query, text, sentence_model):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=250)
    chunks = text_splitter.split_text(text)
    query_embedding = sentence_model.encode(query)
    chunk_embeddings = sentence_model.encode(chunks)
    scores = [query_embedding.dot(chunk) for chunk in chunk_embeddings]
    top_n_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:5]  # Get top 5 most similar chunks
    return [chunks[i] for i in top_n_indices]

def llm_function(query, text, uploaded_file):
    if not text:
        if uploaded_file is not None:
            text = extract_text_from_pdf(uploaded_file)
        else:
            st.error("Please upload a PDF document.")
            return

    relevant_chunks = retrieve_relevant_chunks(query, text, sentence_model)
    relevant_text = " ".join(relevant_chunks)
    prompt = f"Document: {relevant_text} Question: {query} Answer:"
    response = model.generate_content(prompt)

    with st.chat_message("assistant"):
        st.markdown(response.text)

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": response.text})

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
query = st.chat_input("What's your question about the document?")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    llm_function(query, None, uploaded_file)
