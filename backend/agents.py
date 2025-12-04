# backend/agents.py
"""
Multi-agent orchestration for LearnTube (CareerNinja).
Switched to Groq for chat/analysis completions.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

# --- Import New OpenAI Client ---
import openai
from openai import OpenAI

from backend.scraper import scrape_profile
from backend.memory import save_interaction, get_recent_memory, build_memory_context

# Configure API key from env (Switched to look for GROQ_API_KEY)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    # Changed error handling to reflect new key
    pass 

# --- Initialize Groq Client Globally ---
client = None
try:
    if GROQ_API_KEY:
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1" # <--- CRITICAL: Groq API Endpoint
        )
except Exception:
    logger.error("Failed to initialize Groq client.")
    client = None
    
# Model choices (Switched to a Groq Model)
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.1-8b-instant")
MAX_TOKENS = int(os.getenv("ANALYZE_MAX_TOKENS", "900"))

# logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# ----------------------------------------


def _synthesize_job_description(job_title: Optional[str]) -> str:
    """
    Synthesizes a short canonical job description.
    """
    if not job_title:
        return "A generic professional role; focus on transferable skills: communication, problem-solving, teamwork, and role-relevant competencies."
    title = job_title.strip()
    return (
        f"Canonical responsibilities and skills for the role '{title}':\n"
        "- Core responsibilities: deliver outcomes relevant to the role, collaborate cross-functionally, and manage stakeholder communication.\n"
        "- Common skills: relevant technical skills for the title, domain knowledge, problem-solving, communication, and measurable achievements.\n"
        "- Typical deliverables: project outcomes, metrics/KPIs, and domain-specific examples (e.g., product launches, model accuracy improvements, revenue impact).\n"
    )


def _extract_structured_from_model(text: str) -> Dict[str, Any]:
    """
    Attempt to parse structured fields from the model text.
    """
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        candidate = text[start:end]
        parsed = json.loads(candidate)
        return parsed
    except Exception:
        return {
            "match_score": None,
            "recommendations": [],
            "rewritten_sections": {}
        }


async def analyze_profile(linkedin_url: str, target_job: Optional[str] = None, user_id: str = "anon") -> Dict[str, Any]:
    """
    High-level orchestration for profile analysis.
    """
    try:
        # 1. Scrape profile (AWAIT)
        profile = await scrape_profile(linkedin_url)
    except Exception as e:
        logger.exception("Failed to scrape profile")
        raise RuntimeError(f"Failed to scrape LinkedIn profile: {e}")

    # 2. Build canonical job description (synthesized)
    job_desc = _synthesize_job_description(target_job)

    # 3. Save a quick memory that an analysis was requested
    try:
        save_interaction(
            user_id,
            text=f"Requested analysis for URL: {linkedin_url}, target_job: {target_job or 'none'}",
            metadata={"type": "analysis_request", "url": linkedin_url, "target_job": target_job}
        )
    except Exception:
        logger.warning("Failed to save initial analysis_request memory (continuing)")

    # 4. Retrieve recent memory & build relevant memory context
    try:
        memory_context = build_memory_context(user_id, query=target_job or "career profile", top_k=6)
    except Exception:
        memory_context = ""
        logger.warning("Failed to fetch/build memory context")

    # 5. Build prompt for Groq
    system_prompt = (
        "You are an expert career coach and LinkedIn profile optimizer. "
        "Given a user's LinkedIn profile (as JSON) and a canonical job description, "
        "produce a structured JSON object containing:\n"
        "    - match_score: integer 0-100 representing how well the profile matches the job. **CRITICAL INSTRUCTION**: Base this score on a weighted, three-part rubric:\n"
        "      1. **Experience Relevance (Max 40 points):** Assess how closely the duration and relevance of past job titles/descriptions align with the target role's core responsibilities.\n"
        "      2. **Skill Overlap (Max 30 points):** Score the explicit presence and demonstrated use of essential, role-specific skills required by the job description.\n"
        "      3. **Quantifiable Impact (Max 30 points):** Evaluate the presence of measurable achievements, metrics, and KPIs (e.g., 'increased X by Y%'). A profile lacking quantification should score near 0 in this category.\n"
        "    - recommendations: list of 5-8 actionable recommendations (short sentences) to improve the score for this specific job.\n"
        "    - rewritten_sections: object with keys 'headline', 'about', 'experience' (experience can be a list of rewritten bullets)\n"
        "    - notes: any short plain-language explanation (string) of the score or analysis approach.\n\n"
        "Important: After any plain text commentary, include a valid JSON object only (no extra commentary after the JSON). "
        "If you can't compute a score, set match_score to null.\n"
    )

    # append memory context if available (kept short)
    if memory_context:
        system_prompt += "\n\nUSER MEMORY CONTEXT (most relevant):\n" + memory_context + "\n\n"

    user_prompt = (
        "PROFILE_JSON:\n"
        f"{json.dumps(profile, ensure_ascii=False)}\n\n"
        "JOB_DESCRIPTION:\n"
        f"{job_desc}\n\n"
        "INSTRUCTIONS:\n"
        "1) Provide an overall short analysis (2-3 sentences), then output the JSON object described above.\n"
        "2) In rewritten_sections.headline: provide a one-line headline optimized for LinkedIn with keywords.\n"
        "3) In rewritten_sections.about: provide an 80-120 word About summary, active voice, quantify achievements if possible.\n"
        "4) In rewritten_sections.experience: return an array of rewritten bullet points (3-6) summarizing key achievements from the Experience section aligned to the job.\n"
        "Be concise and actionable.\n"
    )

    # 6. Call Groq Completion (NEW SYNTAX)
    if client is None:
        raise RuntimeError("Groq client not initialized.")
        
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            n=1
        )
        analysis_text = response.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("Groq request failed")
        raise RuntimeError(f"Groq request failed: {e}")

    # 7. Parse JSON object from model output
    parsed = _extract_structured_from_model(analysis_text)

    # Normalize fields with fallbacks
    match_score = parsed.get("match_score")
    try:
        if isinstance(match_score, (int, float)):
            match_score = float(match_score)
            if match_score < 0: match_score = 0.0
            if match_score > 100: match_score = 100.0
        else:
            match_score = None
    except Exception:
        match_score = None

    recommendations = parsed.get("recommendations") or []
    rewritten = parsed.get("rewritten_sections") or {}

    # 8. Save analysis summary and rewritten content into memory
    try:
        save_interaction(
            user_id,
            text=f"Analysis result for {linkedin_url}: score={match_score}; recs={recommendations}",
            metadata={"type": "analysis_result", "match_score": match_score, "target_job": target_job}
        )
        if isinstance(rewritten, dict):
            for key, value in rewritten.items():
                if isinstance(value, list):
                    text_val = "\n".join(value)
                else:
                    text_val = str(value)
                save_interaction(
                    user_id,
                    text=f"rewritten_{key}: {text_val}",
                    metadata={"type": "rewritten", "section": key, "target_job": target_job}
                )
    except Exception:
        logger.warning("Failed to persist analysis/rewrite memories")

    # 9. Build final result dict
    result = {
        "profile": profile,
        "analysis_text": analysis_text,
        "match_score": match_score,
        "recommendations": recommendations,
        "rewritten_sections": rewritten
    }

    return result


# CHANGED TO ASYNC
async def chat_agent(user_id: str, message: str) -> str:
    """
    Handles conversational chat, retrieving memory context.
    """
    
    # 1. Retrieve RECENT and RELEVANT memory for context
    better_query = f"Based on the user's most recent profile analysis and the message: {message}. Focus on the last target job and match score."
    
    try:
        memory_context = build_memory_context(user_id, query=better_query, top_k=8)
    except Exception:
        memory_context = ""
        logger.warning("Failed to fetch/build memory context for chat.")

    # 2. Build the prompt
    system_prompt = (
        "You are LearnTube, a concise and supportive AI career coach. "
        "Your goal is to provide actionable career guidance, skill gap analysis, "
        "and advice on optimizing a LinkedIn profile. "
        "IMPORTANT: Always prioritize the most RECENT analysis results and REWRITTEN SECTIONS found in the memory context. "
        "If the user is asking about job titles, ensure they align with the last targeted job or the skills in the last profile review. "
        "If no relevant analysis is found in memory, state that the profile must be analyzed first."
    )
    
    user_prompt = f"USER MESSAGE: {message}\n"
    if memory_context:
        user_prompt = "RELEVANT MEMORY CONTEXT:\n" + memory_context + "\n\n" + user_prompt
    
    # 3. Call Groq Completion (NEW SYNTAX)
    if client is None:
        return "I'm sorry, the AI service is currently unavailable due to an initialization error."

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=250, # Keep chat responses snappy
            temperature=0.6,
            n=1
        )
        assistant_reply = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq chat request failed: {e}")
        assistant_reply = "I'm sorry, I seem to be having trouble processing your request right now. Please try again later."
    
    # 4. Save interaction
    try:
        # Save user message
        save_interaction(user_id, text=message, metadata={"type": "chat_user"})
        # Save assistant reply
        save_interaction(user_id, text=assistant_reply, metadata={"type": "chat_assistant"})
    except Exception:
        logger.warning("Failed to save chat interaction to memory (continuing)")
        
    return assistant_reply


# If this module is run directly, a simple smoke-test could be placed here.
if __name__ == "__main__":
    import asyncio
    
    demo_url = "https://www.linkedin.com/in/sample-profile"
    
    async def demo_run():
        print("--- Running Analyze Profile Demo (via Groq) ---")
        try:
            out = await analyze_profile(demo_url, target_job="Product Manager", user_id="demo_user")
            print(json.dumps(out, ensure_ascii=False, indent=2))
        except Exception as e:
            print("Analyze run failed:", e)
        
        print("\n--- Running Chat Agent Demo (via Groq) ---")
        try:
            chat_response = await chat_agent(user_id="demo_user", message="What was my match score and what are the top 3 recommendations?")
            print("Assistant:", chat_response)
        except Exception as e:
            print("Chat run failed:", e)
            
    try:
        asyncio.run(demo_run())
    except Exception as e:
        print("Demo initialization failed:", e)