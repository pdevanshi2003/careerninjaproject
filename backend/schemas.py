# cd ~# backend/schemas.py

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class AnalyzeRequest(BaseModel):
    linkedin_url: str
    target_job_title: Optional[str] = None
    user_id: Optional[str] = None


class ChatMessage(BaseModel):
    user_id: str
    message: str


class AnalyzeResponse(BaseModel):
    profile: Dict[str, Any]
    analysis_text: str
    match_score: Optional[float]
    recommendations: List[str]
    rewritten_sections: Dict[str, Any]