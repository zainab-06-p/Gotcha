# app.py — HuggingFace Spaces entry point
import subprocess, sys
subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"])
