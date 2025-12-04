# backend/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv 
import os 

# --- NEW: Load environment variables from the .env file immediately ---
load_dotenv() 
# --------------------------------------------------------------------

from backend.schemas import AnalyzeRequest, ChatMessage
from backend.agents import analyze_profile, chat_agent

app = FastAPI(title="CareerNinja LearnTube API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Analyze LinkedIn profile and generate match score,
    recommendations, and rewritten sections.
    """
    try:
        # AWAIT the async analyze_profile function
        result = await analyze_profile(
            linkedin_url=req.linkedin_url,
            target_job=req.target_job_title,
            user_id=req.user_id or "anon"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(req: ChatMessage):
    """
    Handle an interactive chat message, using memory for context.
    """
    try:
        # AWAIT the async chat_agent function
        response_text = await chat_agent(
            user_id=req.user_id or "anon",
            message=req.message
        )
        # Return a simple dict response
        return {"user_id": req.user_id or "anon", "message": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))