---
title: HealthCareDiagnosisLLM
emoji: ⚡
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.43.2
app_file: app.py
pinned: false
license: apache-2.0
short_description: HackSLU submission
---

# AI Diagnostic Assistant

This project is an AI-powered diagnostic assistant that provides preliminary medical assessments based on user-input symptoms and audio analysis.

---

## Installation
```bash
# Clone repository
git clone https://github.com/Omkar31415/HealthCareBot_LLM.git
cd HealthCareBot_LLM

# Create virtual environment
conda create -p venv python==3.10 -y

# Install dependencies
pip install -r requirements.txt
```
## Instructions to run

Uncomment these commands in physical_health.py and mental_health.py
os.getenv("KEYS") 

Run the App
```
streamlit run app.py
```
.env file should have:
```
HUGGINGFACE_API_KEY="your_hf_key_here"
GROQ_API_KEY="your_groq_key_here"
ANTHROPIC_API_KEY="your_claude_key_here"
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
