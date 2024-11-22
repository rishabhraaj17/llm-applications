import os
import logging
import bs4
from langchain.chains import RetrievalQA
from langchain_openai.llms import OpenAI
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the library or framework URL
docs_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"

# Load the documents from the library or framework
def load_docs(url):
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
        )
    ),
            )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        return texts
    except Exception as e:
        logging.error(f"Error loading documents from URL {url}: {e}")
        return []

# Create a vector store from the loaded documents
def create_vector_store(texts):
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text", 
                                      base_url=f"{os.environ.get('OLLAMA_HOST')}:{os.environ.get('OLLAMA_PORT')}")
        vector_store = FAISS.from_documents(texts, embeddings)
        return vector_store
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        return None

# Set up the QA system with the vector store and OpenAI model
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
        return None

# Main function to run the RAG pipeline
def main():
    texts = load_docs(docs_url)
    if not texts:
        logging.error("No documents loaded. Exiting.")
        return
    
    vector_store = create_vector_store(texts)
    if not vector_store:
        logging.error("Failed to create vector store. Exiting.")
        return
    
    qa = setup_qa(vector_store)
    if not qa:
        logging.error("Failed to set up QA system. Exiting.")
        return
    
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        try:
            response = qa.invoke(query)
            print(response)
        except Exception as e:
            logging.error(f"Error processing query '{query}': {e}")

if __name__ == "__main__":
    main()