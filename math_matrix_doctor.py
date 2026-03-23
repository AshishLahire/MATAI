import os
import sys
import pytesseract
from PIL import Image
import whisper
import torch
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from core.rag_pipeline import get_rag

load_dotenv()

def check_env():
    print("--- 📝 Checking Environment ---")
    keys = ["GROQ_API_KEY", "TESSERACT_PATH"]
    for key in keys:
        val = os.getenv(key)
        if val:
            print(f"✅ {key} is set.")
        else:
            print(f"❌ {key} is MISSING.")

def check_tesseract():
    print("\n--- 📷 Checking Tesseract OCR ---")
    tesseract_path = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract Version: {version}")
    except Exception as e:
        print(f"❌ Tesseract Error: {e}")

def check_whisper():
    print("\n--- 🎤 Checking Whisper ASR ---")
    try:
        model = whisper.load_model("tiny") # Use tiny for quick check
        print(f"✅ Whisper model loaded successfully. Device: {model.device}")
    except Exception as e:
        print(f"❌ Whisper Error: {e}")

def check_groq():
    print("\n--- 🤖 Checking Groq API ---")
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile")
        res = llm.invoke("Hello, are you online?")
        print(f"✅ Groq Response: {res.content[:50]}...")
    except Exception as e:
        print(f"❌ Groq Error: {e}")

def check_rag():
    print("\n--- 📚 Checking RAG (FAISS) ---")
    try:
        rag = get_rag()
        res = rag.query("calculus")
        if res["documents"]:
            print(f"✅ RAG is working. Found {len(res['documents'])} docs.")
        else:
            print("⚠️ RAG initialized but no documents found for 'calculus'. Did you run init_rag.py?")
    except Exception as e:
        print(f"❌ RAG Error: {e}")

if __name__ == "__main__":
    check_env()
    check_tesseract()
    check_whisper()
    check_groq()
    check_rag()
