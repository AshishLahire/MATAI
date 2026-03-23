from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
import json
import os
from dotenv import load_dotenv
import sympy

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"

class AgentState(TypedDict):
    input_text: str
    structured_problem: Dict[str, Any]
    topic: str
    rag_context: List[str]
    solution: str
    verification: Dict[str, Any]
    explanation: str
    needs_hitl: bool
    confidence_level: str # Low, Medium, High
    agent_trace: List[str]

def parser_node(state: AgentState):
    llm = ChatGroq(model=MODEL_NAME)
    prompt = f"""You are a Math Problem Parser. Extract structured information from the input.
    If the input is OCR text, it might have errors. Fix obvious typos.
    
    Input: {state['input_text']}
    
    Return ONLY a JSON object with:
    {{
      "problem_text": "Cleaned version of the problem",
      "variables": ["list", "of", "variables"],
      "constants": ["list", "of", "constants"],
      "missing_info": "Description of any missing details",
      "is_ambiguous": boolean (true if the problem is unclear or has OCR gibberish)
    }}
    """
    response = llm.invoke(prompt)
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
    except:
        data = { "problem_text": state['input_text'], "is_ambiguous": True, "variables": [], "constants": [] }
    
    trace = state.get("agent_trace", [])
    trace.append(f"Parser: Extracted {len(data.get('variables', []))} variables. Ambiguous: {data.get('is_ambiguous')}")
    
    return {
        "structured_problem": data,
        "needs_hitl": data.get("is_ambiguous", False),
        "agent_trace": trace
    }

def router_node(state: AgentState):
    llm = ChatGroq(model=MODEL_NAME)
    problem = state['structured_problem']['problem_text']
    prompt = f"Categorize this math problem into ONE: algebra, calculus, geometry, probability, or logic. Reply with just the word. Problem: {problem}"
    response = llm.invoke(prompt)
    topic = response.content.strip().lower().strip('.')
    
    trace = state.get("agent_trace", [])
    trace.append(f"Router: Classified as {topic}.")
    
    return {"topic": topic, "agent_trace": trace}

def solver_node(state: AgentState):
    llm = ChatGroq(model=MODEL_NAME)
    context = "\n".join(state.get('rag_context', []))
    problem = state['structured_problem']['problem_text']
    topic = state['topic']
    
    prompt = f"""You are an Expert Math Solver. Solve the problem step-by-step.
    
    Context from Knowledge Base:
    {context}
    
    Problem: {problem}
    Topic: {topic}
    
    Instructions:
    1. Show every step clearly.
    2. Use LaTeX for all mathematical expressions (e.g., $x^2$).
    3. If applicable, conclude with 'Final Answer: <answer>'.
    4. If the problem is algebraic, ensure symbols are defined.
    """
    response = llm.invoke(prompt)
    solution = response.content
    
    # Optional: Internal Sympy check for simple expressions if possible
    trace = state.get("agent_trace", [])
    if "final answer:" in solution.lower():
        trace.append("Solver: Generated solution with explicit final answer.")
    else:
        trace.append("Solver: Generated step-by-step solution.")
    
    return {"solution": solution, "agent_trace": trace}

def verifier_node(state: AgentState):
    llm = ChatGroq(model=MODEL_NAME)
    problem = state['structured_problem']['problem_text']
    solution = state['solution']
    
    prompt = f"""You are a Math Verifier. Review the solution for the given problem.
    
    Problem: {problem}
    Solution: {solution}
    
    Check for:
    1. Calculation errors.
    2. Logical inconsistencies.
    3. Completeness.
    
    Return ONLY a JSON object:
    {{
      "is_correct": boolean,
      "confidence": float (0.0 to 1.0),
      "critique": "Specific feedback on why it is correct or incorrect",
      "suggested_fix": "Optional fix if incorrect"
    }}
    """
    response = llm.invoke(prompt)
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
    except:
        data = {"is_correct": False, "confidence": 0, "critique": "Verification engine error."}
    
    conf_val = data.get('confidence', 0)
    conf_str = "High" if conf_val > 0.8 else "Medium" if conf_val > 0.5 else "Low"
    
    trace = state.get("agent_trace", [])
    trace.append(f"Verifier: Checked solution. Confidence: {conf_str}. Correct: {data.get('is_correct')}")
    
    return {
        "verification": data,
        "confidence_level": conf_str,
        "needs_hitl": not data.get('is_correct') or conf_val < 0.7,
        "agent_trace": trace
    }

def explainer_node(state: AgentState):
    llm = ChatGroq(model=MODEL_NAME)
    # The explainer creates a student-friendly tutorial based on the solution and verification
    critique = state['verification'].get('critique', '')
    prompt = f"""You are a Math Tutor. Present the final solution to a student.
    
    Original Solution:
    {state['solution']}
    
    Verification Feedback:
    {critique}
    
    Output:
    1. A friendly summary.
    2. The step-by-step solution (formatted in clean LaTeX).
    3. A 'Key Takeaway' or 'Concept' section.
    """
    response = llm.invoke(prompt)
    
    trace = state.get("agent_trace", [])
    trace.append("Explainer: Formatted tutorial for student.")
    
    return {"explanation": response.content, "agent_trace": trace}

def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("parser", parser_node)
    workflow.add_node("router", router_node)
    workflow.add_node("solver", solver_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("explainer", explainer_node)
    
    workflow.set_entry_point("parser")
    workflow.add_edge("parser", "router")
    workflow.add_edge("router", "solver")
    workflow.add_edge("solver", "verifier")
    workflow.add_edge("verifier", "explainer")
    workflow.add_edge("explainer", END)
    
    return workflow.compile()
