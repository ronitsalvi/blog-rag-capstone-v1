from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .models import QuestionRequest, QuestionResponse
from .config import get_config
from .retrieval import search_and_answer

config = get_config()

app = FastAPI(
    title="Blog Q&A Chatbot",
    description="LMS Blog Q&A system with local search and web fallback",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config['app']['cors_origins'],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        response = await search_and_answer(request.question, request.top_k)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )