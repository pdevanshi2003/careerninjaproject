# careerninjaproject

CareerNinja – AI-Powered LinkedIn Profile & Career Assistant

An AI-driven, interactive system designed to analyze LinkedIn profiles, generate job-fit insights, rewrite profile content, and provide personalized career guidance using a multi-agent architecture and persistent memory.

🚀 Features:
1. LinkedIn Profile Scraping using Apify
2. Profile Quality Analysis (About, Experience, Skills, Gaps)
3. Job Fit Evaluation with role-based match scoring
4. ATS-Optimized Content Rewriting
5. Skill Gap Detection & Career Recommendations
6. Chat-Based Interactive UI via Streamlit
7. Session + Persistent Memory using LangGraph checkpointer
8. Multi-Agent Architecture for modular and scalable functionality

📁 Project Structure:

1. app.py – FastAPI backend server
2. streamlit_app.py – Streamlit-based user interface
3. agents.py – Multi-agent system handling profile analysis, job-fit scoring, and content rewriting
4. memory.py – Memory system implementation with checkpointing and context retention
5. schemas.py – Pydantic models for validating inputs and outputs
6. scraper.py – Apify LinkedIn Scraper integration for extracting profile data
7. requirements.txt – List of all project dependencies
8. README.md – Main project documentation and setup guide

🛠️ Setup Instructions:

1. Clone the repository
git clone <your_repo_url>
cd <project_folder>

2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Set up environment variables
Create a .env file:
APIFY_TOKEN=your_apify_token
OPENAI_API_KEY=your_openai_key


5. Run the FastAPI backend
uvicorn app:app --reload --port 8000

6. Run the Streamlit UI
streamlit run streamlit_app.py

7. Open the app in browser
http://localhost:8005

🧠 Core Components
1. Backend — FastAPI
2. API endpoints for scraping, analysis, chat, job fit, and rewriting.
3. Connects agents with frontend.
4. Manages authentication and memory routing.
5. Agents — LangGraph Multi-Agent System
6. Scraper Agent
7. Profile Analysis Agent
8. Job Fit Agent
9. Content Rewrite Agent
10. Memory Agent
11. Frontend — Streamlit
12. Chat-based UI
13. Input box for LinkedIn URL
14. Displays match scores, recommendations, and rewritten content
15. Memory System
16. Short-term (chat session)
17. Long-term (vector DB + checkpointing)

