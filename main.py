from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import shutil
from core.multimodal import get_processor
from core.rag_pipeline import get_rag
from core.memory import get_memory
from agents.workflow import create_graph

app = FastAPI(title="Math Mentor Backend API")

# Models
class SolveRequest(BaseModel):
    text: str

# Endpoints
@app.post("/solve")
async def solve_problem(request: SolveRequest):
    graph = create_graph()
    rag = get_rag()
    memory = get_memory()
    
    # RAG Retrieval
    rag_results = rag.query(request.text)
    context = rag_results.get("documents", [])
    
    # Agent Workflow
    inputs = {
        "input_text": request.text,
        "rag_context": context,
        "agent_trace": []
    }
    
    final_state = await graph.ainvoke(inputs)
            
    # Store in Memory
    if final_state:
        memory.store_interaction(
            request.text, 
            final_state.get('structured_problem', {}), 
            final_state.get('solution', ''), 
            final_state.get('verification', {})
        )
    
    return {
        "text": request.text,
        "context": context,
        "state": final_state
    }

@app.post("/process/image")
async def process_image(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    processor = get_processor()
    result = processor.process_image(temp_path)
    os.remove(temp_path)
    return result

@app.post("/process/audio")
async def process_audio(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    processor = get_processor()
    result = processor.process_audio(temp_path)
    os.remove(temp_path)
    return result

@app.get("/history")
async def get_history():
    memory = get_memory()
    return memory.get_history()

@app.get("/")
def read_root():
    return {"status": "Math Mentor API is Online"}
