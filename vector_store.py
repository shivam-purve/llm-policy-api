from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def build_vector_store(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

def search_vector_store(query, vectorstore, k=5):
    return vectorstore.similarity_search(query, k=k)
