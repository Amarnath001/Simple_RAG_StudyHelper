# 📚 RAG Study Helper

A simple yet powerful RAG (Retrieval-Augmented Generation) application designed to help students learn, study, and practice using their study materials. This app uses AI to provide personalized tutorials, answer questions, generate practice questions, and create study notes from your uploaded documents.

## Features

- **💬 Q&A**: Ask any question about your study materials and get AI-powered answers with source citations
- **📖 Tutorials**: Generate step-by-step tutorials on any topic from your study materials
- **❓ Practice Questions**: Get practice questions to test your understanding (customizable by topic and number)
- **📝 Study Notes**: Generate concise study notes and summaries for quick review
- **📁 Document Management**: Upload and manage multiple PDF, TXT, or MD files
- **🔍 Smart Retrieval**: Uses vector embeddings to find the most relevant content from your materials

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Gemini API key:**
   
   Create a `.env` file in the project root:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-1.5-flash  # Optional: specify model name (default: gemini-1.5-flash)
   ```
   
   Get your free API key from: https://makersuite.google.com/app/apikey
   
   **Available models**: `gemini-1.5-flash` (recommended, free), `gemini-1.5-pro`, `gemini-pro`

## Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Upload your study materials:**
   - Click on the sidebar
   - Upload PDF, TXT, or MD files containing your study materials
   - Click "Load Documents" to process them

3. **Use the features:**
   - **Q&A Tab**: Ask questions about your materials
   - **Tutorials Tab**: Generate tutorials on specific topics
   - **Practice Questions Tab**: Get practice questions
   - **Study Notes Tab**: Generate study notes and summaries

## Project Structure

```
Simple_RAG_StudyHelper/
├── app.py                 # Main Streamlit application
├── rag_system.py          # RAG system implementation
├── document_processor.py  # Document loading and processing
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## How It Works

1. **Document Processing**: Uploaded documents are split into chunks for efficient retrieval
2. **Embedding**: Text chunks are converted to vector embeddings using local models (sentence-transformers) - **free, no API calls needed!**
3. **Vector Store**: Embeddings are stored in ChromaDB for fast similarity search
4. **Retrieval**: When you ask a question, the system finds the most relevant chunks
5. **Generation**: The LLM (Google Gemini Pro) generates responses based on the retrieved context

**Note**: Embeddings run locally (free), only text generation uses the Gemini API.

## Supported File Formats

- PDF (`.pdf`)
- Text files (`.txt`)
- Markdown (`.md`)

## Tips for Best Results

- Upload well-structured documents with clear headings
- Use specific questions for better answers
- For tutorials, specify the exact topic you want to learn
- Generate practice questions after studying a section
- Use study notes for quick review before exams

## Troubleshooting

**Error: "GEMINI_API_KEY not found"**
- Make sure you've created a `.env` file with your Gemini API key
- Check that the `.env` file is in the project root directory
- Get your API key from: https://makersuite.google.com/app/apikey

**Error: "API quota exceeded"**
- Gemini free tier has rate limits (requests per minute)
- Wait a few minutes and try again
- Check your usage at: https://makersuite.google.com/app/apikey

**Error: "Vector store not initialized"**
- Make sure you've uploaded and loaded documents first
- Click "Load Documents" after uploading files

**Slow responses:**
- The first query may be slower as the system initializes
- Large documents may take longer to process initially

## Future Enhancements

Potential features to add:
- Support for more file formats (Word, PowerPoint, etc.)
- Multiple LLM provider options (OpenAI, Anthropic, local models)
- Conversation history
- Export functionality for generated content
- Multi-language support
- Custom chunking strategies

## License

This project is open source and available for educational purposes.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

---

**Happy Studying! 📖✨**