# mental_health_support_rag.py

import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# LangChain imports for ChromaDB
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import Runnable


# --- Configuration ---
# Define the directory where your mental health documents (PDFs, TXTs) are stored.
DOCS_DIR = 'docs'
# Define the path where the ChromaDB vector store will be saved.
CHROMA_DB_PATH = './chroma_db_mental_health'
# Define the path for the CSV dataset file.
CSV_DATA_PATH = 'mental_health_qa.csv'

# --- 1. Load Environment Variables ---
# Load environment variables from the .env file. This is crucial for securely
# accessing your Hugging Face API token.
load_dotenv()
# Get the Hugging Face API token from environment variables.
hf_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Check if the API token is loaded successfully.
if hf_api_token:
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_api_token
    print('✅ Hugging Face API Token loaded successfully')
else:
    print('❌ HUGGINGFACEHUB_API_TOKEN not found in .env file. Please add it.')
    # Exit or raise an error if the API token is not found, as it's required.
    exit("Hugging Face API token is missing.")

# --- 2. Document Loading and Processing ---
def load_documents() -> List[str]:
    """
    Loads text content from CSV files in the 'docs' directory
    and a specified CSV file.

    Returns:
        List[str]: A list of strings, where each string is the text content
                   from a document or a combined Q&A pair from the CSV.
    """
    doc_texts = []

    # Create docs directory if it doesn't exist
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f'📁 Created {DOCS_DIR} directory')

    print("\n--- Loading Documents ---")

    # Load CSV file for Q&A pairs
    if os.path.exists(CSV_DATA_PATH):
        try:
            df = pd.read_csv(CSV_DATA_PATH)
            df.columns = [col.lower().strip() for col in df.columns]
            # Ensure 'question' and 'answer' columns exist
            if 'question' in df.columns and 'answer' in df.columns:
                # Combine question and answer into a single text for each row
                for index, row in df.iterrows():
                    # Convert to string to handle potential non-string types in CSV
                    combined_text = f"Question: {str(row['question'])}\nAnswer: {str(row['answer'])}"
                    doc_texts.append(combined_text)
                print(f'✅ Loaded CSV: {os.path.basename(CSV_DATA_PATH)} with {len(df)} Q&A pairs.')
            else:
                print(f'❌ CSV file "{CSV_DATA_PATH}" missing "question" or "answer" columns.')
        except Exception as e:
            print(f'❌ Error loading CSV {os.path.basename(CSV_DATA_PATH)}: {e}')
    else:
        print(f'⚠️ CSV file "{CSV_DATA_PATH}" not found. Please ensure it exists.')


    print(f'\n📚 Total raw documents loaded: {len(doc_texts)}')
    return doc_texts

# --- 3. Create or Load ChromaDB Vector Store ---
def get_vector_store(doc_texts: List[str]):
    """
    Creates or loads a ChromaDB vector store from provided document texts.
    If the database already exists, it loads it. Otherwise, it creates a new one.

    Args:
        doc_texts (List[str]): A list of text documents to be used for embedding.

    Returns:
        Chroma: The ChromaDB vector store object.
    """
    print("\n--- Setting up Vector Store ---")

    # Initialize embeddings model. 'all-MiniLM-L6-v2' is a good general-purpose
    # sentence transformer model suitable for many tasks.
    print('🤖 Initializing embedding model (SentenceTransformerEmbeddings)...')
    embeddings_model = SentenceTransformerEmbeddings(
        model_name='all-MiniLM-L6-v2'
    )

    # Check if a persistent ChromaDB already exists
    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        print(f'🔍 Loading existing ChromaDB from: {CHROMA_DB_PATH}')
        # Load the existing database
        db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings_model,
            collection_name='mental_health_docs'
        )
        print('✅ ChromaDB loaded successfully!')
    else:
        if not doc_texts:
            print('❌ No documents to process. Cannot create ChromaDB.')
            return None

        # Split documents into chunks. This is crucial for RAG, as LLMs have context windows.
        # Smaller chunks allow more relevant information to be retrieved.
        # chunk_size: maximum characters in a chunk.
        # chunk_overlap: characters to overlap between chunks to maintain context.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        all_chunks = []
        print(f'📝 Splitting {len(doc_texts)} raw documents into smaller chunks...')

        # Process each raw document and split it into chunks.
        for i, text in enumerate(doc_texts):
            chunks = text_splitter.split_text(text)
            # Create Document objects from the text chunks for LangChain compatibility.
            # This allows metadata to be added later if needed.
            all_chunks.extend([Document(page_content=chunk) for chunk in chunks])
            print(f'  📄 Document {i+1} yielded {len(chunks)} chunks.')

        if all_chunks:
            # Create ChromaDB vector store from the chunks.
            # This step embeds the text chunks and stores them in the database.
            print(f'🔍 Building new ChromaDB from {len(all_chunks)} chunks...')
            db = Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings_model,
                persist_directory=CHROMA_DB_PATH,
                collection_name='mental_health_docs'
            )
            # Persistence is now handled automatically by Chroma when using persist_directory
            # db.persist() # Removed this line
            print(f'✅ New ChromaDB created and saved successfully!')
            print(f'💾 Database saved to: {CHROMA_DB_PATH}')
        else:
            print('❌ No text chunks generated. ChromaDB not created.')
            db = None
    return db

# --- 4. Create RAG Chain for Question Answering ---
def get_rag_chain(db):
    """
    Creates and returns a Retrieval-Augmented Generation (RAG) chain
    using ChromaDB and a Hugging Face language model.
    """
    print("\n--- Setting up RAG Chain ---")

    if db is None:
        print('❌ Cannot create RAG chain - vector database is not available.')
        return None

    # ✅ Initialize the correct Hugging Face LLM using HuggingFaceHub
    print("💡 Initializing Hugging Face LLM via HuggingFaceHub...")
    try:
        '''llm = HuggingFaceEndpoint(
            repo_id="https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            temperature = 0.5,
            max_new_tokens = 512
        )'''
        llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-base",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            temperature=0.5,
            max_new_tokens=512
        )
    except Exception as e:
        print(f"❌ Failed to initialize HuggingFaceHub model: {e}")
        return None

    # ✅ Define prompt template
    prompt_template = """
{context}

Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # ✅ Create document chain
    print("📝 Creating document chain...")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Custom Retriever that selects only the 3rd document
    class ThirdOnlyRetriever(Runnable):
        def __init__(self, retriever):
            self.retriever = retriever

        def invoke(self, input, config=None):
            query_text = input["input"] if isinstance(input, dict) else input
            docs = self.retriever.invoke(query_text)
            if len(docs) >= 3:
                return [docs[2]]
            elif docs:
                return [docs[-1]]
            else:
                return [Document(page_content="No relevant context found.")]

    # Required for LangChain to track config
        def with_config(self, **kwargs):
            return self

    # ✅ Create retriever
    print("🔎 Creating retriever from ChromaDB...")
    original_retriever = db.as_retriever(search_kwargs={"k": 3})
    wrapped_retriever = ThirdOnlyRetriever(original_retriever)

    # ✅ Create final RAG chain
    print("🔗 Creating final RAG chain...")
    rag_chain = create_retrieval_chain(wrapped_retriever, question_answer_chain)

    print("✅ RAG chain created successfully!")
    return rag_chain

# --- Main execution block for testing or initial setup ---
# This block will run when mental_health_support_rag.py is executed directly.
# It's useful for verifying the setup before running the Streamlit app.
if __name__ == "__main__":
    print("--- Running Mental Health RAG Setup Script ---")

    # Load all documents (PDFs, TXTs, CSV)
    documents = load_documents()

    # Create or load the vector store
    vector_store = get_vector_store(documents)

    # Create the RAG chain
    chain = get_rag_chain(vector_store)

    if chain:
        print("\n--- Testing RAG Chain ---")
        try:
            test_query = "What are some common symptoms of depression?"
            print(f'🧪 Testing with query: "{test_query}"')
            response = chain.invoke({'input': test_query})
            print(f'\n💬 Response: {response}')
        except Exception as e:
            import traceback
            print(f'❌ An error occurred during test invocation: {e}')

    else:
        print("Initialization failed. Cannot run test query.")

