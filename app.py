import streamlit as st
import os
import requests
import time
from PIL import Image
from dotenv import load_dotenv

# Setup
load_dotenv()
st.set_page_config(page_title="Math Matrix AI - Production", layout="wide", page_icon="📐")

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Global Theme */
    .stApp { 
        background: radial-gradient(circle at top right, #1e293b, #0f172a); 
        color: #f1f5f9; 
    }
    
    /* Typography */
    .main-title { 
        font-size: 3.5rem; 
        font-weight: 800; 
        background: linear-gradient(90deg, #60a5fa, #c084fc); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 0.5rem;
    }
    
    /* Cards & Containers */
    .glass-card { 
        background: rgba(30, 41, 59, 0.4); 
        border: 1px solid rgba(148, 163, 184, 0.1); 
        border-radius: 16px; 
        padding: 1.5rem; 
        backdrop-filter: blur(16px); 
        margin-bottom: 1.5rem; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Agent Trace Steps */
    .agent-step { 
        padding: 1rem; 
        border-radius: 12px; 
        background: rgba(15, 23, 42, 0.5); 
        border-left: 4px solid #3b82f6; 
        margin-bottom: 0.75rem; 
        font-size: 0.95rem;
        transition: transform 0.2s ease;
    }
    .agent-step:hover {
        transform: translateX(4px);
        background: rgba(30, 41, 59, 0.8);
    }
    
    /* HITL Alert */
    .hitl-alert {
        padding: 1rem;
        border-radius: 12px;
        background: rgba(234, 179, 8, 0.1);
        border: 1px solid rgba(234, 179, 8, 0.3);
        border-left: 4px solid #eab308;
        color: #fef08a;
        margin-bottom: 1rem;
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: 1px solid rgba(255,255,255,0.05);
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.2);
        border-bottom: 2px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">📐 Math Matrix AI</h1>', unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>Your Intelligent Multimodal Math Mentor.</p>", unsafe_allow_html=True)

tabs = st.tabs(["📝 Text Input", "📷 Image Input", "🎤 Audio Input", "🧠 Brain (Memory)"])

if 'process_data' not in st.session_state:
    st.session_state.process_data = None
if 'rag_context' not in st.session_state:
    st.session_state.rag_context = None

def run_pipeline(input_text):
    with st.status("🚀 Initializing Math Matrix Core...", expanded=True) as status:
        try:
            st.write("Analysing input and routing to agents...")
            time.sleep(0.5) # UX delay
            response = requests.post(f"{API_URL}/solve", json={"text": input_text})
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.process_data = data['state']
                st.session_state.rag_context = data['context']
                
                # Check if backend flagged this for HITL
                needs_hitl = data['state'].get('needs_hitl', False)
                if needs_hitl:
                    status.update(label="⚠️ Human Review Required!", state="error", expanded=True)
                    st.error("The agents detected low confidence or ambiguity in the problem.")
                else:
                    status.update(label="✨ Full Solution Generated!", state="complete", expanded=False)
            else:
                status.update(label="❌ Backend Error", state="error", expanded=True)
                st.error("Backend Error: " + response.text)
        except Exception as e:
            status.update(label="❌ Connection Error", state="error", expanded=True)
            st.error(f"Could not connect to backend at {API_URL}. Ensure main.py is running.")

# --- TABS ---
with tabs[0]:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    txt_input = st.text_area("Type your math problem here:", height=120, placeholder="e.g., Calculate the derivative of f(x) = x^2 * sin(x)")
    if st.button("Solve Text", type="primary", use_container_width=True): 
        run_pipeline(txt_input)
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Image of a math problem", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_file, use_column_width=True, caption="Uploaded Image")
        with col2:
            if st.button("Extract & Solve", type="primary", use_container_width=True):
                with st.spinner("Extracting text with Tesseract OCR..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    res = requests.post(f"{API_URL}/process/image", files=files)
                    
                    if res.status_code == 200:
                        ocr_data = res.json()
                        ocr_text = ocr_data.get('text', '')
                        confidence = ocr_data.get('confidence', 0)
                        
                        st.write(f"**Extracted Text:**\n```\n{ocr_text}\n```")
                        
                        if ocr_data.get('needs_hitl', False) or confidence < 0.6:
                            st.markdown(f"<div class='hitl-alert'>⚠️ Low OCR Confidence ({confidence:.2f}). Please review and edit the text before solving.</div>", unsafe_allow_html=True)
                            edited_text = st.text_area("Edit text:", value=ocr_text, height=100, key="img_edit")
                            if st.button("Confirm & Solve", key="img_solve_btn"):
                                run_pipeline(edited_text)
                        else:
                            st.success(f"High Confidence Extraction ({confidence:.2f})")
                            run_pipeline(ocr_text)
                    else:
                        st.error("Failed to process image.")
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])
    
    st.markdown("<p style='text-align: center; margin: 0.5rem 0; color: #94a3b8; font-weight: 500;'>OR</p>", unsafe_allow_html=True)
    recorded_audio = st.audio_input("Record Live Audio")
    
    final_audio = audio_file or recorded_audio
    if final_audio:
        if audio_file:
            st.audio(audio_file)
            
        if st.button("Transcribe & Solve", type="primary", use_container_width=True):
            with st.spinner("Transcribing with Whisper ASR..."):
                files = {"file": (final_audio.name, final_audio.getvalue())}
                res = requests.post(f"{API_URL}/process/audio", files=files)
                
                if res.status_code == 200:
                    asr_data = res.json()
                    asr_text = asr_data.get('text', '')
                    
                    st.write(f"**Transcript:**\n```\n{asr_text}\n```")
                    
                    if asr_data.get('needs_hitl', False):
                        st.markdown("<div class='hitl-alert'>⚠️ Audio unclear or too short. Please review.</div>", unsafe_allow_html=True)
                        edited_audio_text = st.text_area("Edit text:", value=asr_text, height=100, key="aud_edit")
                        if st.button("Confirm & Solve", key="aud_solve_btn"):
                            run_pipeline(edited_audio_text)
                    else:
                        st.success("Transcription Successful")
                        run_pipeline(asr_text)
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[3]:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🧠 Long-Term Memory")
    if st.button("🔄 Refresh Memory"):
        pass # Streamlit reruns on button click anyway
        
    try:
        res = requests.get(f"{API_URL}/history")
        if res.status_code == 200:
            history = res.json()
            if not history:
                st.info("No interactions stored yet.")
            for item in history:
                timestamp_str = item.get('timestamp', 'Unknown Time')[:16].replace('T', ' ')
                prob_preview = item.get('input_text', '')[:60] + "..."
                with st.expander(f"🕒 {timestamp_str} | {prob_preview}"):
                    st.markdown("**Original Input:**")
                    st.info(item.get('input_text', ''))
                    st.markdown("**Final Solution:**")
                    st.markdown(item.get('solution', 'No solution stored.'))
                    verif = item.get('verifier', {})
                    if verif:
                        st.caption(f"Verification: {'Correct' if verif.get('is_correct') else 'Incorrect'} (Conf: {verif.get('confidence', 0)})")
    except:
        st.info("Backend offline or memory file missing.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- RESULTS SECTION ---
if st.session_state.process_data:
    st.divider()
    res = st.session_state.process_data
    
    col_main, col_sidebar = st.columns([2.5, 1])
    
    with col_main:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### ✨ Final Solution & Tutorial")
        
        needs_hitl = res.get('needs_hitl', False)
        if needs_hitl:
            st.warning("⚠️ The Math Verifier flagged this solution for review. See Agent Trace for details.")
            
        st.markdown(res.get('explanation', res.get('solution', 'No solution found.')))
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_sidebar:
        st.markdown("<div class='glass-card' style='padding: 1rem;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0'>🤖 Agent Trace</h4>", unsafe_allow_html=True)
        
        # Display agent actions chronologically
        for step in res.get('agent_trace', []):
            st.markdown(f'<div class="agent-step">{step}</div>', unsafe_allow_html=True)
            
        # Display Confidence Overall
        conf_level = res.get('confidence_level', 'Unknown')
        color = "green" if conf_level == "High" else "orange" if "Medium" in conf_level else "red"
        st.markdown(f"<p style='margin-top: 1rem; font-weight: 600;'>System Confidence: <span style='color: {color};'>{conf_level}</span></p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
