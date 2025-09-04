# Blog Q&A Chatbot

An advanced RAG (Retrieval-Augmented Generation) chatbot that answers questions using local blog content with LLM-enhanced responses.

## Features

- **RAG Architecture**: Combines vector similarity search with LLM answer generation
- **LLM Integration**: Powered by Gemini 1.5 Flash for intelligent responses
- **Local Blog Search**: Vector similarity search through 955+ blog articles
- **Clean Text Processing**: HTML-to-text conversion with noise removal
- **Interactive Web UI**: Modern Gradio interface with provider selection
- **Multiple LLM Providers**: Support for Gemini and OpenAI APIs
- **REST API**: FastAPI server with comprehensive endpoints

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### 2. Environment Setup
Configure your API keys:
```bash
cp .env.example .env
# Edit .env and add your API keys:
# GEMINI_API_KEY=your_gemini_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here  # Optional
```

### 3. Data Setup
Place your `v2-all-blogs-extracted.xlsx` file in the `Documentation/` folder, then:
```bash
python3 convert_v2_excel.py
python3 manual_index_builder.py
```

### 4. Start Server

**Option A: Gradio Web Interface (Recommended)**
```bash
python3 gradio_app.py
```
Visit `http://localhost:7861` for the interactive web interface.

**Option B: FastAPI Server**
```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## Usage

### Web Interface (Gradio)
1. Open `http://localhost:7861` in your browser
2. Select your preferred LLM provider (Gemini Pro or OpenAI GPT-3.5)
3. Type your question in the text box
4. Click "🔍 Ask Question" or press Enter
5. Get LLM-enhanced answers with blog citations

**Features:**
- 🤖 LLM-powered intelligent responses
- 🔄 Multiple AI provider support
- 💡 Sample questions to get started
- 🔗 Clickable source URLs with proper citations
- 🎨 Clean, responsive interface
- ⚡ Fast retrieval with vector search

### Terminal Testing
```bash
# Basic question
curl -X POST "http://localhost:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'

# Formatted output
curl -X POST "http://localhost:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How to use pandas?"}' | python3 -m json.tool

# Health check
curl -X GET "http://localhost:8001/healthz"
```

### Browser Testing
Visit `http://localhost:8001/docs` for interactive Swagger UI

## API Response Format

```json
{
  "answer": "LLM-generated intelligent answer with proper citations...",
  "source_urls": [
    {
      "title": "Blog Post Title",
      "url": "https://360digitmg.com/blog/post-url"
    }
  ],
  "provider_used": "gemini",
  "fallback_used": false
}
```

## Configuration

### Environment Variables (.env)
- `GEMINI_API_KEY`: Required for Gemini LLM integration
- `OPENAI_API_KEY`: Optional for OpenAI GPT integration
- `LOG_LEVEL`: Logging level (default: INFO)

### System Configuration (config.yaml)
- **LLM Settings**: `llm.default_provider`, `llm.temperature`, `llm.max_tokens`
- **Retrieval Settings**: `retrieval.min_local_relevance` (default: 0.36)
- **Chunk size**: `retrieval.chunk_size` (default: 600)

## Project Structure

```
app/
├── main.py              # FastAPI application
├── models.py            # Data models and schemas
├── llm_providers.py     # LLM integration (Gemini, OpenAI)
├── retrieval.py         # RAG pipeline with vector search
├── config.py            # Configuration management
├── utils_text.py        # Text processing utilities
└── storage/             # Data storage (excluded from git)
    ├── v2_blog_data.parquet
    └── manual_vector_index.pkl
gradio_app.py            # Web UI application
config.yaml             # System configuration
.env                    # Environment variables (API keys)
```

## System Requirements

- Python 3.8+
- 8GB+ RAM (for embedding generation)
- ~4GB disk space (for vector index)

## Technology Stack

- **Architecture**: RAG (Retrieval-Augmented Generation)
- **LLM Integration**: Google Gemini 1.5 Flash, OpenAI GPT-3.5 
- **API**: FastAPI + Uvicorn
- **Web UI**: Gradio with provider selection
- **Vector Search**: Sentence-transformers (all-MiniLM-L6-v2)
- **Text Processing**: NLTK, BeautifulSoup
- **Data**: Pandas, PyArrow

Built with modern RAG architecture combining retrieval and generation for intelligent, citation-backed responses.