@echo off
REM StudyMate Launcher Script for Windows

echo 🚀 Starting StudyMate - Your AI Study Buddy
echo ==========================================

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  .env file not found. Please create one with your OPENAI_API_KEY.
    echo    Copy .env and add your OpenAI API key.
    pause
    exit /b 1
)

REM Check if OPENAI_API_KEY is set
findstr /C:"OPENAI_API_KEY=your_openai_api_key_here" .env >nul
if %errorlevel% equ 0 (
    echo ⚠️  Please set your OPENAI_API_KEY in the .env file.
    pause
    exit /b 1
) else (
    echo ✅ API key found in .env
)

REM Activate the existing rag-venv
echo 🐍 Activating rag-venv...
call ..\rag-venv\Scripts\activate.bat

REM Start Streamlit app
echo 🎯 Starting StudyMate...
echo 📱 Open your browser to http://localhost:8501
echo.
streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.runOnSave true

pause