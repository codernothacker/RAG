from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os

class VectorStore:
    def __init__(self, persist_directory: str = "data/chroma"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
    def add_documents(self, documents):
        self.db.add_documents(documents)
        self.db.persist()
        
    def similarity_search(self, query: str, k: int = 4):
        return self.db.similarity_search(query, k=k)
