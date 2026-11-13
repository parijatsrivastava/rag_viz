@echo off
REM Starts the RAG Part 1 Visualizer and opens it in your browser.
REM To stop it, close this window.
cd /d "%~dp0"
start "" http://localhost:8501
".venv\Scripts\streamlit.exe" run app.py
pause
