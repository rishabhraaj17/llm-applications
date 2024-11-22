import os
import logging
import streamlit as st
import bs4
from langchain.chains import RetrievalQA
from langchain_openai.llms import OpenAI
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)


# Function to load documents from a URL
def load_docs(url):
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                    # class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        return texts
    except Exception as e:
        logging.error(f"Error loading documents from URL {url}: {e}")
        st.error(f"Error loading documents from URL {url}: {e}")
        return []

# Function to create a vector store from loaded documents
def create_vector_store(texts):
    try:
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text", 
            base_url=f"{os.environ.get('OLLAMA_HOST')}:{os.environ.get('OLLAMA_PORT')}"
        )
        vector_store = FAISS.from_documents(texts, embeddings)
        return vector_store
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        st.error(f"Error creating vector store: {e}")
        return None

# Function to set up the QA system with the vector store and OpenAI model
def setup_qa(vector_store):
    try:
        llm = OpenAI(
            model="hf:meta-llama/Meta-Llama-3.1-405B-Instruct"
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
        )
        return qa
    except Exception as e:
        logging.error(f"Error setting up QA system: {e}")
        st.error(f"Error setting up QA system: {e}")
        return None

# Streamlit app
def main():
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    st.title("RAG Chatbot")

    # Initialize session state variables
    if 'qa' not in st.session_state:
        st.session_state.qa = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'docs_url' not in st.session_state:
        st.session_state.docs_url = ''

    # URL input section
    st.header("Knowledge Base Setup")
    docs_url = st.text_input("Enter the URL of the knowledge base:", 
                              value=st.session_state.docs_url, 
                              key='url_input')
    st.session_state.docs_url = docs_url

    # Load Knowledge Base Button
    if st.button("Load Knowledge Base"):
        if not docs_url:
            st.error("Please enter a valid URL.")
        else:
            with st.spinner("Loading documents and setting up QA system..."):
                try:
                    # Load docs
                    texts = load_docs(docs_url)
                    vector_store = create_vector_store(texts)
                    st.session_state.qa = setup_qa(vector_store)
                    
                    # Clear previous chat history
                    st.session_state.chat_history = []
                    
                    st.success("Knowledge base loaded and QA system set up successfully!")
                except Exception as e:
                    logging.error(f"Error loading documents or setting up QA system: {e}")
                    st.error(f"Error loading documents or setting up QA system: {e}")

    # Chat Interface
    st.header("Chat Interface")

    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    # Check if QA system is ready
    if st.session_state.qa is not None:
        # Chat input
        prompt = st.chat_input("Enter your query")
        
        if prompt:
            # Add user message to chat history
            st.session_state.chat_history.append({
                'role': 'user', 
                'content': prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process query
            with st.chat_message("assistant"):
                with st.spinner("Processing query..."):
                    try:
                        # Simulate processing time
                        time.sleep(1)
                        
                        # Get response from QA system
                        response = st.session_state.qa.invoke(prompt)
                        response = response['result']
                        
                        # Display response
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({
                            'role': 'assistant', 
                            'content': response
                        })
                    
                    except Exception as e:
                        error_message = f"Error processing query: {e}"
                        st.error(error_message)
                        
                        # Add error to chat history
                        st.session_state.chat_history.append({
                            'role': 'assistant', 
                            'content': error_message
                        })
    else:
        st.info("Please load a knowledge base first.")

# Run the app
if __name__ == "__main__":
    main()