# Blog Q&A Chatbot

A FastAPI-based chatbot that answers questions using local blog content with vector similarity search.

## Features

- **Local Blog Search**: Vector similarity search through 955+ blog articles
- **Clean Text Processing**: HTML-to-text conversion with noise removal
- **REST API**: FastAPI server with `/ask` endpoint
- **Extractive Answers**: Returns relevant excerpts with source URLs
- **Interactive UI**: Built-in Swagger documentation at `/docs`

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### 2. Data Setup
Place your `v2-all-blogs-extracted.xlsx` file in the `Documentation/` folder, then:
```bash
python3 convert_v2_excel.py
python3 manual_index_builder.py
```

### 3. Start Server
```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## Usage

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
  "answer": "Detailed answer from blog content...",
  "excerpt": "Blog excerpt with key information...",
  "source": {
    "title": "Blog Post Title",
    "url": "https://360digitmg.com/blog/post-url",
    "relevance": 0.85
  },
  "fallback_used": false
}
```

## Configuration

Edit `config.yaml` to customize:
- **Relevance threshold**: `retrieval.min_local_relevance` (default: 0.36)
- **Chunk size**: `retrieval.chunk_size` (default: 600)
- **Results count**: `retrieval.top_k` (default: 5)

## Project Structure

```
app/
├── main.py              # FastAPI application
├── models.py            # Data models and schemas
├── retrieval.py         # Vector search and answer composition
├── config.py            # Configuration management
├── utils_text.py        # Text processing utilities
└── storage/             # Data storage (excluded from git)
    ├── v2_blog_data.parquet
    └── manual_vector_index.pkl
```

## System Requirements

- Python 3.8+
- 8GB+ RAM (for embedding generation)
- ~4GB disk space (for vector index)

## Technology Stack

- **API**: FastAPI + Uvicorn
- **Vector Search**: Sentence-transformers (all-MiniLM-L6-v2)
- **Text Processing**: NLTK, BeautifulSoup
- **Data**: Pandas, PyArrow

Built with extractive summarization for accurate, citation-backed responses.