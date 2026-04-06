import streamlit as st
from openai import OpenAI
import time

# --- Personal Project Settings ---
st.set_page_config(
    page_title="My Local NLP Toolkit | Project S11",
    page_icon="✍️",
    layout="wide"
)

# --- Clean Dark Mode UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Outfit:wght@500;700&display=swap');
    
    :root {
        --bg-color: #0e1117;
        --card-bg: #161b22;
        --border-color: #30363d;
        --accent-color: #58a6ff;
        --text-primary: #c9d1d9;
        --text-secondary: #8b949e;
    }

    .main {
        background-color: var(--bg-color);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    .stTextArea textarea {
        background-color: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        font-size: 1rem;
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: var(--accent-color);
        color: white;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        border: none;
    }

    .stButton>button:hover {
        background-color: #1f6feb;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.2);
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        color: #ffffff !important;
        letter-spacing: -0.01em;
    }

    .phase-card {
        background-color: var(--card-bg);
        padding: 2.2rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin-bottom: 20px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: My Config ---
with st.sidebar:
    st.markdown("### Settings for my project")
    ollama_url = st.text_input("Where is Ollama running?", value="http://localhost:11434/v1")
    model_name = st.text_input("Which model to use?", value="llama3.2:1b")
    
    st.markdown("---")
    st.markdown("### Project Notes")
    st.write("I built this to run everything locally on my own machine using Ollama and a three-step process I saw in class (intent, content, and polish).")

# --- Backend Logics ---
client = OpenAI(base_url=ollama_url, api_key="ollama")

def talk_to_llm(system_msg, user_msg):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Wait, something went wrong here: {str(e)}"

# --- Main Interface ---
st.title("My Local Text Analysis Tool")
st.markdown("##### Final project for the Speech and Language processing course")

my_input = st.text_area("Put your text here", height=250, placeholder="Paste whatever you want me to analyze...")

# --- Processing Pipeline ---
if st.button("Start the analysis process"):
    if not my_input:
        st.error("I can't analyze nothing! Please write something first.")
    else:
        # Start the three-phase pipeline visualization
        with st.status("I'm working on it...", expanded=True) as status:
            
            # PHASE 1: Intent
            st.write("First, I'm trying to figure out what this text is trying to say...")
            intent_msg = (
                "Identify the primary intent and target audience of the following text. "
                "Keep it concise and professional. Do not use emojis."
            )
            intent_result = talk_to_llm(intent_msg, my_input)
            
            # PHASE 2: Core
            st.write("Now, I'm pulling out the most important bits and summarizing them...")
            core_msg = (
                "Provide a detailed summary and extract 3 key bullet points from the text. "
                "Ensure maximum information density. Formal tone."
            )
            core_result = talk_to_llm(core_msg, my_input)
            
            # PHASE 3: Refine
            st.write("Finally, I'm putting it all together in a nice, professional report...")
            refinement_msg = (
                "Combine the provided intent and synthesis into a single, polished executive report. "
                "Ensure a very formal and professional tone. Remove any phrases like 'Here is your report'. "
                "Structure it clearly."
            )
            final_report = talk_to_llm(refinement_msg, f"Intent: {intent_result}\n\nSynthesis: {core_result}")
            
            status.update(label="All done!", state="complete", expanded=False)

        # Output Display
        st.markdown("---")
        st.markdown("### The Final Result")
        
        tab1, tab2 = st.tabs(["Refined Report", "Step-by-step trace"])
        
        with tab1:
            st.markdown(f"""
            <div class="phase-card">
                {final_report}
            </div>
            """, unsafe_allow_html=True)
            
        with tab2:
            st.markdown("#### Intent Analysis")
            st.info(intent_result)
            st.markdown("#### Main Synthesis")
            st.info(core_result)

st.markdown("---")
st.caption("Developed by me for the NLP course (S11)")
