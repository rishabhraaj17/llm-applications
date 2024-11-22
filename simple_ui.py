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

    st.title("RAG (Retriever-Augmented Generation) Pipeline")

    # Initialize session state variables if they don't exist
    if 'qa' not in st.session_state:
        st.session_state.qa = None
    if 'docs_url' not in st.session_state:
        st.session_state.docs_url = ''

    # User input for URL
    docs_url = st.text_input("Enter the URL of the knowledge base:", 
                              value=st.session_state.docs_url, 
                              key='url_input')
    st.session_state.docs_url = docs_url

    # Button to load documents and set up the QA system
    if st.button("Load Knowledge Base"):
        if not docs_url:
            st.error("Please enter a valid URL.")
        else:
            with st.spinner("Loading documents and setting up QA system..."):
                try:
                    logging.debug("Loading documents...")
                    # Load docs
                    texts = load_docs(docs_url)
                    vector_store = create_vector_store(texts)
                    st.session_state.qa = setup_qa(vector_store)
                    logging.debug("QA system initialized.")
                    st.success("Knowledge base loaded and QA system set up successfully!")
                except Exception as e:
                    logging.error(f"Error loading documents or setting up QA system: {e}")
                    st.error(f"Error loading documents or setting up QA system: {e}")

    # Allow user to enter a query only if the QA system is set up
    if st.session_state.qa is not None:
        # Maintain query state using session_state
        if 'query' not in st.session_state:
            st.session_state.query = ''

        # Query input with preserved state
        query = st.text_input("Enter your query:", 
                               value=st.session_state.query, 
                               key='query_input')
        st.session_state.query = query

        # Process query button
        if st.button("Get Answer"):
            if query:  # Only process the query if it's not empty
                with st.spinner("Processing query..."):
                    try:
                        logging.debug(f"Query received: {query}")
                        # Simulate time delay to show spinner
                        time.sleep(1)  # Simulate processing time
                        response = st.session_state.qa.invoke(query)  # Call the QA system
                        logging.debug(f"Response: {response}")
                        if response:
                            st.write(response)
                        else:
                            st.warning("No answer found for your query.")
                        
                        # Clear the query after processing
                        st.session_state.query = ''
                    except Exception as e:
                        logging.error(f"Error processing query '{query}': {e}")
                        st.error(f"Error processing query '{query}': {e}")
            else:
                st.warning("Please enter a query before clicking 'Get Answer'.")

# Run the app
if __name__ == "__main__":
    main()