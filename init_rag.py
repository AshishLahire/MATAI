from core.rag_pipeline import get_rag
import os

def init_db():
    rag = get_rag()
    kb_path = "knowledge_base/math_essentials.md"
    
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Split by section for chunking
            sections = content.split("\n# ")
            docs = []
            metadatas = []
            
            for i, section in enumerate(sections):
                if not section.strip(): continue
                docs.append(section)
                metadatas.append({"source": "math_essentials", "section": i})
            
            rag.add_documents(docs, metadatas)
            print("Knowledge base initialized with FAISS.")

if __name__ == "__main__":
    init_db()
