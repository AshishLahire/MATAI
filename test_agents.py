import pytest
import asyncio
from agents.workflow import create_graph

def test_agent_graph_simple_math():
    graph = create_graph()
    inputs = {
        "input_text": "What is 2 + 2?",
        "rag_context": [],
        "agent_trace": []
    }
    
    async def run_graph():
        return await graph.ainvoke(inputs)
        
    final_state = asyncio.run(run_graph())
            
    assert final_state is not None
    assert "solution" in final_state
    assert "explanation" in final_state
    assert final_state["structured_problem"]["problem_text"] != ""
    assert final_state["confidence_level"] in ["Low", "Medium", "High"]
    
def test_agent_graph_ambiguous_math():
    graph = create_graph()
    inputs = {
        "input_text": "Solve x?+*3=y what",
        "rag_context": [],
        "agent_trace": []
    }
    
    async def run_graph():
        return await graph.ainvoke(inputs)
        
    final_state = asyncio.run(run_graph())
            
    assert final_state is not None
    assert final_state.get("needs_hitl", False) == True
