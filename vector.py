from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

load_dotenv()

pdfs = [
    "./data/20240716890312078.pdf"
]

db_location = "./chroma_langchain_db"


def pdfs_to_sqlite_embeddings(pdf_paths, sqlite_db_path):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    vector_store = Chroma(
        collection_name="pdf_embeddings",
        persist_directory=sqlite_db_path,
        embedding_function=embeddings
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=200
    )

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        chunks = text_splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata["source"] = pdf_path

        vector_store.add_documents(chunks)

        print(f"✅ Added embeddings for {pdf_path}")

    print(f"\n📦 Embeddings stored in: {sqlite_db_path}")


# ✅ Only run embedding if DB does NOT exist
if not os.path.exists(db_location) or len(os.listdir(db_location)) == 0:
    print("📥 Creating vector database...")
    pdfs_to_sqlite_embeddings(pdfs, db_location)
else:
    print("✅ Using existing vector database")


# ✅ Load DB
vector_store = Chroma(
    collection_name="pdf_embeddings",
    persist_directory=db_location,
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large")
)

vector_retriever = vector_store.as_retriever(search_kwargs={"k": 15})

loader = PyPDFLoader(pdfs[0])
docs = loader.load()
bm25 = BM25Retriever.from_documents(docs)

hybrid = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=[0.4, 0.6]  # BM25: 40%, Embeddings: 60%
)
hybrid = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=[0.4, 0.6]  # BM25: 40%, Embeddings: 60%
)
llm = ChatOllama(
    model="llama3.2", 
    temperature=0
)

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=hybrid
)