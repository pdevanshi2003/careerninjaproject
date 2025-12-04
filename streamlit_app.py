import streamlit as st
import requests
import os

# NOTE: Using port 8001 as per your previous error resolution
API_BASE = st.secrets.get('API_BASE', 'http://localhost:8005')

# Set a fixed user ID for the demo session for persistent memory access
CURRENT_USER_ID = 'devanshi' 

# Page config
st.set_page_config(page_title='LearnTube by CareerNinja', layout='wide')

# Custom CSS --- FIX APPLIED HERE ---
try:
    # 1. Open the CSS file
    with open('frontend/styles.css') as f:
        css_content = f.read()
        
        # 2. Wrap the CSS content in <style> tags before rendering
        # This ensures the browser interprets it as style rules, not visible text.
        st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
        
except FileNotFoundError:
    st.warning("Could not find frontend/styles.css. Ensure it's in the correct path.")
# --- END FIX ---


with st.container():
    st.markdown('<div class="topbar"><h1>LearnTube — CareerNinja</h1></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:
        linkedin_url = st.text_input('Paste LinkedIn profile URL', '')
    with col2:
        target = st.text_input('Target job title (optional)', '')
    analyze = st.button('Analyze')

if analyze and linkedin_url:
    with st.spinner('Analyzing profile...'):
        resp = requests.post(f'{API_BASE}/analyze', json={
            'linkedin_url': linkedin_url,
            'target_job_title': target,
            'user_id': CURRENT_USER_ID
        })
        if resp.status_code == 200:
            data = resp.json()
            # Basic layout: profile overview | match score | recommendations | rewritten
            st.subheader('Profile Overview')
            st.json(data.get('profile'))

            st.subheader('AI Analysis & Recommendations')
            st.markdown(data.get('analysis_text'))

            st.subheader('Rewritten Sections')
            for k,v in (data.get('rewritten_sections') or {}).items():
                st.markdown(f'**{k}**')
                st.write(v)
        else:
            st.error(f'Error: {resp.text}')

# --- NEW/MODIFIED CHAT INTERFACE ---
st.sidebar.title('Chat with LearnTube')
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

msg = st.sidebar.text_input('Message')

if st.sidebar.button('Send') and msg:
    # 1. Save user message to session state
    st.session_state['messages'].append({'from':'user','text':msg})
    
    # 2. Call the backend chat endpoint
    with st.spinner('Thinking...'):
        chat_resp = requests.post(f'{API_BASE}/chat', json={
            'user_id': CURRENT_USER_ID,
            'message': msg
        })

    if chat_resp.status_code == 200:
        data = chat_resp.json()
        assistant_reply = data.get('message', '(assistant) I received an empty response.')
        # 3. Save assistant reply to session state
        st.session_state['messages'].append({'from':'assistant','text':assistant_reply})
    else:
        st.error(f'Chat Error: {chat_resp.status_code} - {chat_resp.text}')
        st.session_state['messages'].append({'from':'assistant','text':'(assistant) Error communicating with the backend chat agent.'})
    
    # Rerun the script to update the message display
    st.rerun()


# Display messages (in reverse order, showing latest at the top)
for m in st.session_state['messages'][::-1]:
    if m['from']=='assistant':
        st.sidebar.markdown(f"**Assistant:** {m['text']}")
    else:
        st.sidebar.markdown(f"**You:** {m['text']}")