#!/bin/bash
# StudyMate Launcher Script

echo "🚀 Starting StudyMate - Your AI Study Buddy"
echo "=========================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Please create one with your OPENAI_API_KEY."
    echo "   Copy .env and add your OpenAI API key."
    exit 1
fi

# Check if OPENAI_API_KEY is set
if ! grep -q "OPENAI_API_KEY=your_openai_api_key_here" .env; then
    echo "✅ API key found in .env"
else
    echo "⚠️  Please set your OPENAI_API_KEY in the .env file."
    exit 1
fi

# Activate the existing rag-venv
echo "🐍 Activating rag-venv..."
source ../rag-venv/Scripts/activate

# Start Streamlit app
echo "🎯 Starting StudyMate..."
echo "📱 Open your browser to http://localhost:8501"
echo ""
streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.runOnSave true