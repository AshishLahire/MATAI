import pytesseract
import whisper
import numpy as np
from PIL import Image
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Configure Tesseract path from .env or default
tesseract_cmd = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

class MultimodalProcessor:
    def __init__(self):
        # Initialize Whisper model lazily to save resources
        self.asr_model = None

    def _get_whisper(self):
        if self.asr_model is None:
            self.asr_model = whisper.load_model("base")
        return self.asr_model

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image using Tesseract OCR."""
        try:
            image = Image.open(image_path)
            # Use Tesseract for OCR
            extracted_text = pytesseract.image_to_string(image)
            
            # Basic analysis to determine confidence and HITL need
            text_clean = extracted_text.strip()
            word_count = len(text_clean.split())
            
            # If text is too short or contains many non-alphanumeric chars, it implies low confidence
            special_chars = sum(1 for c in text_clean if not c.isalnum() and not c.isspace())
            
            needs_hitl = False
            confidence = 0.9
            
            if word_count < 2:
                needs_hitl = True
                confidence = 0.3
            elif special_chars > len(text_clean) * 0.3: # Too much noise
                needs_hitl = True
                confidence = 0.4
                
            return {
                "text": text_clean,
                "confidence": confidence,
                "needs_hitl": needs_hitl
            }
        except Exception as e:
            return {"text": "", "error": str(e), "confidence": 0.0, "needs_hitl": True}

    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """Convert speech to text using Whisper."""
        try:
            model = self._get_whisper()
            result = model.transcribe(audio_path)
            text = result.get("text", "").strip()
            
            # Whisper usually provides confident transcripts if audible
            needs_hitl = len(text) < 5
            
            return {
                "text": text,
                "confidence": 0.9 if not needs_hitl else 0.4,
                "needs_hitl": needs_hitl
            }
        except Exception as e:
            return {"text": "", "error": str(e), "confidence": 0.0, "needs_hitl": True}

def get_processor():
    return MultimodalProcessor()
