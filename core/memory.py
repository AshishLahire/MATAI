import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

class MathMemory:
    def __init__(self, file_path: str = "data/memory.jsonl"):
        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def store_interaction(self, input_text: str, structured: Dict, solution: str, verifier: Dict, feedback: Optional[Dict] = None):
        data = {
            "timestamp": datetime.now().isoformat(),
            "input_text": input_text,
            "structured": structured,
            "solution": solution,
            "verifier": verifier,
            "feedback": feedback
        }
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    def get_history(self, limit: int = 20) -> List[Dict]:
        if not os.path.exists(self.file_path):
            return []
        
        history = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if len(history) >= limit: break
                    if not line.strip(): continue
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return history

    def search_similar(self, query: str, limit: int = 3) -> List[Dict]:
        history = self.get_history(limit=100)
        # Simple keyword match for demo production
        results = []
        query_words = set(query.lower().split())
        for item in history:
            text = item.get("input_text", "").lower()
            if any(word in text for word in query_words):
                results.append(item)
            if len(results) >= limit: break
        return results

def get_memory():
    return MathMemory()
