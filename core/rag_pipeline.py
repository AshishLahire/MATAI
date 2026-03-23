import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from typing import List, Dict, Any

class MathRAG:
    def __init__(self, persist_directory: str = "data/faiss_index"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.persist_directory):
            self.vector_store = FAISS.load_local(
                self.persist_directory, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            # Initialize empty if needed, but usually we add docs first
            pass

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(documents, self.embeddings, metadatas=metadatas)
        else:
            self.vector_store.add_texts(documents, metadatas=metadatas)
        self.vector_store.save_local(self.persist_directory)

    def query(self, text: str, k: int = 3) -> Dict[str, Any]:
        if self.vector_store is None:
            return {"documents": [], "metadatas": []}
        
        docs = self.vector_store.similarity_search(text, k=k)
        return {
            "documents": [doc.page_content for doc in docs],
            "metadatas": [doc.metadata for doc in docs]
        }

def get_rag():
    return MathRAG()
