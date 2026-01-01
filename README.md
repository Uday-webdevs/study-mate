# StudyMate - Your AI Study Buddy 🤖📚

A student-friendly chatbot that helps you learn from your documents using advanced RAG (Retrieval-Augmented Generation) technology with multi-level fallback systems.

## Features ✨

- **📄 PDF Upload**: Upload your study materials (PDF format)
- **💬 Interactive Chat**: Ask questions about your documents
- **🎯 Smart Retrieval**: Multi-level search with fallback mechanisms
- **🛡️ Safety First**: Built-in safety checks for educational use only
- **📊 Confidence Levels**: Know how reliable each answer is
- **🔄 Persistent Chat**: Conversation history is maintained
- **🎨 Beautiful UI**: Clean, student-friendly interface

## Tech Stack 🛠️

- **Frontend**: Streamlit
- **AI/ML**: OpenAI GPT-4o-mini, OpenAI Embeddings
- **Vector Database**: ChromaDB
- **RAG Framework**: LangChain
- **Document Processing**: PyPDF

## Quick Start 🚀

1. **Navigate to the study-mate directory**
   ```bash
   cd study-mate
   ```

2. **Set up your OpenAI API key**
   - Edit the `.env` file and replace `your_openai_api_key_here` with your actual OpenAI API key

3. **Run the application**
   - **Windows**: Double-click `run.bat` or run `run.bat` in command prompt
   - **Linux/Mac**: Run `chmod +x run.sh && ./run.sh` in terminal

4. **Open your browser** to `http://localhost:8501`

## Manual Setup (Alternative) 📋

If you prefer manual setup:

1. **Activate the virtual environment**
   ```bash
   # Windows
   ..\rag-venv\Scripts\activate

   # Linux/Mac
   source ../rag-venv/bin/activate
   ```

2. **Install dependencies** (already done)
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Usage 📖

1. **Upload Document**: Use the sidebar to upload a PDF document
2. **Wait for Processing**: The app will process and index your document
3. **Start Asking**: Type your questions in the chat interface
4. **Get Answers**: Receive AI-powered responses based on your document

## RAG System Architecture 🏗️

The app implements a sophisticated 5-level fallback RAG system:

1. **Level 1**: Direct vector similarity search
2. **Level 2**: Keyword expansion search
3. **Level 3**: Semantic expansion search
4. **Level 4**: Cross-domain concept search
5. **Level 5**: Graceful fallback with helpful suggestions

## Safety & Security 🔒

- **Educational Focus**: Only accepts academic and study-related queries
- **Content Filtering**: Built-in safety checks prevent misuse
- **Local Processing**: Documents are processed locally (no external uploads)
- **API Key Protection**: Secure environment variable handling

## Configuration ⚙️

Customize the app behavior through environment variables in `.env`:

- `MAX_FILE_SIZE_MB`: Maximum PDF file size (default: 10MB)
- `ALLOWED_EXTENSIONS`: File types allowed (default: pdf)
- `OPENAI_MODEL`: GPT model to use (default: gpt-4o-mini)
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)

## Validation 🧪

Run the setup validator to check everything is working:

```bash
python validate_setup.py
```

## Troubleshooting 🔧

### Common Issues:

1. **"OpenAI API key not found"**
   - Make sure you've added your API key to the `.env` file

2. **"Module not found" errors**
   - Ensure you're using the correct virtual environment
   - Run `pip install -r requirements.txt`

3. **App won't start**
   - Check that port 8501 is not in use
   - Try running `streamlit run app.py --server.port 8502`

4. **Document processing fails**
   - Ensure your PDF is not password-protected
   - Check file size is under the limit

## Project Structure 📁

```
study-mate/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (configure your API key)
├── README.md             # This file
├── validate_setup.py     # Setup validation script
├── run.sh                # Linux/Mac launcher
├── run.bat               # Windows launcher
├── .gitignore           # Git ignore rules
└── chroma_db/           # Vector database storage (created automatically)
```

## Contributing 🤝

This is an educational project. Feel free to suggest improvements or report issues.

## License 📄

Built for educational purposes.

---

**Made with ❤️ for students worldwide**