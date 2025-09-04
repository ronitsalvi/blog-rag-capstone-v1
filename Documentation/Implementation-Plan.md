# Blog Q&A Chatbot Implementation Plan

## Project Overview

Building a production-ready Blog Q&A chatbot for an LMS with three-phase approach:
1. **Phase 1**: Answer from local blog data with excerpts and links
2. **Phase 2**: Web fallback search with competitor blocking and citation logging  
3. **Phase 3**: Topic guardrails for on-topic questions only

## Tech Stack (Lean Implementation)

- **Language**: Python 3.11+
- **API**: FastAPI (JSON responses)
- **Vector Store**: ChromaDB with sentence-transformers
- **Embeddings**: `all-MiniLM-L6-v2` (local, no API costs)
- **Web Search**: `duckduckgo_search` (free)
- **Text Extraction**: `trafilatura` for HTML cleaning
- **Summarization**: Extractive (rule-based sentence selection)
- **Optional**: Ollama integration for abstractive summarization
- **No Docker**: Direct Python deployment

## Project Structure

```
/Documentation
  Implementation-Plan.md
/app
  __init__.py
  main.py              # FastAPI app and routes
  models.py            # Pydantic models and BlogDoc schema
  config.py            # Configuration management
  data_processor.py    # LLM-based SQL processing + HTML cleaning
  build_index.py       # Vector index creation
  retrieval.py         # Local search and answer composition
  web_fallback.py      # Web search and competitor blocking
  guardrails.py        # Topic classification
  blocklist.py         # Domain filtering management
  utils_text.py        # Text processing utilities
  storage/
    blog_data.parquet  # Clean blog data from SQL processing
    blocklist.json     # Competitor domains
    suppressed_links.log # Blocked search results log
    chroma/            # Vector database
/tests
  conftest.py
  test_data_processing.py
  test_retrieval.py
  test_guardrails.py  
  test_blocklist.py
config.yaml
.env.example
requirements.txt
Makefile
README.md
```

## Phase 1: Data Preparation & LLM Processing

### 1.1 LLM-Based SQL Data Extraction

**Approach**: Use LLM to parse `blog.sql` INSERT statements into clean structured data

**Implementation Strategy:**
- Create prompt template for SQL parsing
- Process INSERT statements in batches to handle large file
- Map SQL columns to canonical BlogDoc schema
- Handle complex cases: escaped quotes, NULL values, longtext with embedded HTML

**SQL Column Mapping:**
```python
# From blog.sql schema to BlogDoc
{
    'id': 'id',
    'blog_title': 'title', 
    'base_url': 'url',
    'blog_short_description': 'short_desc',
    'blog_description': 'body',  # Needs HTML cleaning
    'author': 'author',
    'blog_date': 'published_at',  # Parse varchar to datetime
    'meta_title + meta_description': 'tags'  # Extract keywords heuristically
}
```

**Output**: Clean parquet file with ~500-1000 blog records

### 1.2 HTML Text Cleaning Pipeline

**Primary Tool**: `trafilatura` for content extraction
**Fallback**: `bleach` for remaining HTML tags

**Cleaning Strategy:**
```python
def clean_html_to_text(html: str) -> str:
    # 1. Extract main content with trafilatura
    # 2. Remove CSS/JS blocks, style tags
    # 3. Convert to clean text, preserve paragraph breaks
    # 4. Normalize whitespace while keeping sentence boundaries
    # 5. Return readable text suitable for embeddings
```

**Quality Checks:**
- Verify text length vs original HTML
- Check for remaining HTML artifacts
- Ensure readability and coherence

### 1.3 Data Models & Core Classes

**BlogDoc Schema:**
```python
from typing import TypedDict
from datetime import datetime

class BlogDoc(TypedDict):
    id: str
    title: str
    url: str                    # base_url from SQL
    short_desc: str
    body: str                   # cleaned blog_description  
    author: str | None
    tags: list[str]             # derived from meta fields
    published_at: datetime | None
```

**Chunk Metadata:**
```python
class DocumentChunk(TypedDict):
    chunk_id: str
    source_id: str              # hash of id + url
    title: str
    url: str
    content: str
    chunk_index: int
    char_start: int
    char_end: int
```

## Phase 2: Core RAG System

### 2.1 Vector Store Implementation

**ChromaDB Setup:**
```python
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize
client = chromadb.PersistentClient(path="/app/storage/chroma")
collection = client.get_or_create_collection(
    name="blog_chunks",
    embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)
```

**Metadata Schema:**
- `source_id`: unique blog identifier
- `title`: blog title for display
- `url`: permalink for source linking
- `chunk_index`: position within document
- `char_span`: for excerpt extraction

### 2.2 Intelligent Chunking Strategy

**Sentence-Aware Chunking:**
```python
def chunk_document(doc: BlogDoc, target_size: int = 600, overlap: int = 120) -> list[DocumentChunk]:
    # 1. Split into sentences using spaCy/NLTK
    # 2. Group sentences to target token count
    # 3. Add overlap from previous/next chunks
    # 4. Create metadata for each chunk
    # 5. Generate unique chunk IDs
```

**Chunking Considerations:**
- Preserve sentence boundaries (no mid-sentence cuts)
- Maintain context overlap for better retrieval
- Handle edge cases: very short/long paragraphs
- Optimize for embedding quality vs storage efficiency

### 2.3 Retrieval Engine

**Search Logic:**
```python
def search_local(question: str, top_k: int = 5) -> list[SearchHit]:
    # 1. Generate question embedding
    # 2. Vector similarity search in ChromaDB
    # 3. Apply MMR for result diversity
    # 4. Filter by relevance threshold (0.36)
    # 5. Merge chunks from same blog post
    # 6. Return ranked results with metadata
```

**Answer Composition:**
```python
def compose_local_answer(hits: list[SearchHit]) -> dict:
    # 1. Extract best sentences for answer (extractive)
    # 2. Get verbatim excerpt from top chunk
    # 3. Format source with title, URL, relevance score
    # 4. Return structured response
```

### 2.4 FastAPI Application Structure

**Core Routes:**
```python
@app.post("/ask")
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    # 1. Input validation and sanitization
    # 2. Topic guardrails check
    # 3. Local retrieval attempt
    # 4. Web fallback if needed
    # 5. Format and return response

@app.get("/healthz")
async def health_check():
    # Basic health check for deployment

@app.post("/admin/blocklist")
async def manage_blocklist(request: BlocklistRequest):
    # Add/remove domains from competitor blocklist

@app.get("/admin/metrics")
async def get_metrics():
    # Return usage stats, fallback rates, suppressed domains
```

**Request/Response Models:**
```python
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)

class QuestionResponse(BaseModel):
    answer: str
    excerpt: str | None = None
    source: SourceInfo | None = None
    citations: list[Citation] | None = None
    fallback_used: bool
    policy_reason: str | None = None
```

### 2.5 Configuration Management

**config.yaml Structure:**
```yaml
app:
  title: "Blog Q&A Chatbot"
  cors_origins: ["http://localhost:3000"]  # LMS host

retrieval:
  min_local_relevance: 0.36
  top_k: 5
  max_snippet_sentences: 3
  chunk_size: 600
  chunk_overlap: 120

guardrails:
  enabled: true
  min_topic_score: 0.35
  allowed_topics:
    - "data science"
    - "machine learning" 
    - "deep learning"
    - "python programming"
    - "pandas"
    - "numpy"
    - "matplotlib"
    - "statistics"
    - "sql database"
    - "apache spark"
    - "tableau"
    - "power bi"
    - "business analytics"
    - "experiment design"
    - "feature engineering"

web_fallback:
  enabled: true
  provider: "duckduckgo"
  max_results: 8
  timeout_seconds: 10
  
logging:
  level: "INFO"
  format: "json"
```

**Environment Variables:**
```bash
# .env.example
ADMIN_TOKEN=change-me-in-production
STRICT_BLOCKLIST=true
OLLAMA_HOST=  # Optional: http://localhost:11434 for abstractive summarization
LOG_LEVEL=INFO
```

## Complexity Assessment & Risk Mitigation

**Low Complexity:**
- FastAPI setup and basic routing
- ChromaDB integration (well-documented)
- DuckDuckGo search integration
- Environment configuration

**Medium Complexity:**
- HTML text cleaning (multiple edge cases)
- Extractive summarization (sentence ranking)
- Topic classification without paid LLMs
- Domain blocklist implementation

**High Complexity (Mitigated):**
- ~~SQL parsing~~ → **Solved with LLM processing**
- ~~Multi-format data handling~~ → **Single parquet pipeline**
- ~~Docker deployment~~ → **Direct Python deployment**

**Risk Mitigation Strategies:**
1. **SQL Processing**: Use LLM for one-time conversion, avoid runtime parsing
2. **HTML Cleaning**: Multiple fallback strategies (trafilatura → bleach → regex)
3. **Text Quality**: Extensive testing with sample blog content
4. **Performance**: Async FastAPI, efficient vector search, result caching
5. **Reliability**: Comprehensive error handling and graceful degradation

## Development Timeline

**Day 1**: LLM data processing + project structure
**Day 2-3**: Core retrieval system + local search
**Day 4**: Web fallback + competitor blocking  
**Day 5-6**: Guardrails + admin endpoints + testing
**Day 7**: Documentation + deployment preparation

This plan prioritizes reliability and maintainability over complexity, using LLM assistance for the most challenging parsing tasks while keeping the runtime system lean and dependency-minimal.