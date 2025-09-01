from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class BlogDoc(TypedDict):
    id: str
    title: str
    url: str
    short_desc: str
    body: str
    author: str
    tags: List[str]
    published_at: datetime

class DocumentChunk(TypedDict):
    chunk_id: str
    source_id: str
    title: str
    url: str
    content: str
    chunk_index: int
    char_start: int
    char_end: int

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)

class SourceInfo(BaseModel):
    title: str
    url: str
    relevance: float

class Citation(BaseModel):
    title: str
    url: str

class QuestionResponse(BaseModel):
    answer: str
    excerpt: Optional[str] = None
    source: Optional[SourceInfo] = None
    citations: Optional[List[Citation]] = None
    fallback_used: bool
    policy_reason: Optional[str] = None

class SearchHit(TypedDict):
    content: str
    metadata: dict
    relevance: float

class WebResult(TypedDict):
    title: str
    url: str
    content: str
    relevance: float