@echo off
cd /d "%~dp0"
uv run streamlit run footprint_gui.py
pause