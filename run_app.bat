@echo off
cd /d "%~dp0"
echo ========================================================
echo  AWS BEDROCK 6/58 ULTRA LOTTO PREDICTION AGENT
echo  Model: amazon.nova-pro-v1:0 (us-east-1)
echo ========================================================
if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe -m streamlit run app.py
) else (
    python -m streamlit run app.py
)
pause
