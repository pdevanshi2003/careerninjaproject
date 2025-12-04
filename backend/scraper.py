# backend/scraper.py
"""
Apify-based LinkedIn scraper for reliable, compliant data extraction.
The 'wait_for_finish' parameter is removed to fix the ActorClient.call() error.
"""

import os
import asyncio
from apify_client import ApifyClient 

# Get Apify Token from environment
APIFY_API_TOKEN = os.getenv('APIFY_API_TOKEN')
APIFY_ACTOR_ID = "simpleapi/linkedin-profile-scraper" 

async def scrape_profile(url: str) -> dict:
    if not APIFY_API_TOKEN:
        raise EnvironmentError("APIFY_API_TOKEN not set. Cannot use Apify scraper.")

    client = ApifyClient(APIFY_API_TOKEN) 

    run_input = {
        "urls": [url],
        "proxyConfiguration": {"useApifyProxy": True}
    }

    print(f"Starting Apify Actor {APIFY_ACTOR_ID} for URL: {url}...")

    try:
        # Run the Actor and wait for it to finish
        # FIXED: Removed the unsupported 'wait_for_finish' argument
        run = await client.actor(APIFY_ACTOR_ID).call(
            run_input=run_input
        )
        
        # Fetch results from the Actor's dataset
        dataset_items = await client.dataset(run["defaultDatasetId"]).list_items()
        
        if dataset_items and dataset_items.items:
            scraped_data = dataset_items.items[0]
            
            # --- Map Apify Data to Project Schema ---
            profile = {
                "name": scraped_data.get("fullName", scraped_data.get("profileName", "")),
                "headline": scraped_data.get("headline", ""),
                "about": scraped_data.get("summary", ""),
                "experience": [],
                "skills": scraped_data.get("skills", [])
            }

            # Map experience history (Apify provides detailed history)
            experiences_raw = scraped_data.get("experience", [])
            for exp in experiences_raw:
                # FIXED: Calculate the date string first, then append the dictionary cleanly
                date_info = f"{exp.get('startsAt', {}).get('year', '')} - {exp.get('endsAt', {}).get('year', 'Present')}"
                
                profile["experience"].append({
                    "title": exp.get("title", ""),
                    "company": exp.get("companyName", ""),
                    "date": date_info  # Use the calculated variable correctly
                })
                
            return profile

        return {"name": "Apify Failed", "headline": "", "about": "No data returned by Apify Actor.", "experience": [], "skills": []}

    except Exception as e:
        print(f"Apify call failed: {e}")
        return {"name": "Apify Error", "headline": str(e), "about": "Check Apify logs or credits.", "experience": [], "skills": []}