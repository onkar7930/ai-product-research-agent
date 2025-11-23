"""
AI Product Research Agent - Home Page
Create and manage research sessions for product competitive analysis.
"""

import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Product Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database on first load
from utils.database import (
    init_database, create_session, get_all_sessions, get_session,
    save_document, save_vectors, save_insight, update_session_status,
    get_documents, clear_insights, delete_session
)
from utils.scrapers import (
    fetch_app_store_reviews, fetch_play_store_reviews_rss,
    fetch_changelog_content, parse_app_id
)
from utils.text_processing import chunk_text, extract_pain_points
from utils.embeddings import get_embeddings_batch


def check_env_vars():
    """Check if required environment variables are set."""
    required_vars = ["OPENAI_API_KEY", "SUPABASE_HOST", "SUPABASE_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    return missing


def init_app():
    """Initialize the application."""
    missing_vars = check_env_vars()
    if missing_vars:
        st.error(f"⚠️ Missing required environment variables: {', '.join(missing_vars)}")
        st.info("""
        **Setup Instructions:**

        1. **For local development**, create a `.env` file or set environment variables:
        ```
        OPENAI_API_KEY=your_openai_api_key
        SUPABASE_HOST=your_project.supabase.co
        SUPABASE_PASSWORD=your_supabase_password
        SUPABASE_USER=postgres
        SUPABASE_DATABASE=postgres
        SUPABASE_PORT=5432
        ```

        2. **For Streamlit Cloud**, add these in the Secrets management section.

        Get your Supabase credentials from: https://supabase.com/dashboard
        Get your OpenAI API key from: https://platform.openai.com/api-keys
        """)
        return False

    try:
        init_database()
        return True
    except Exception as e:
        st.error(f"⚠️ Database connection error: {str(e)}")
        st.info("Please check your Supabase credentials and ensure the database is accessible.")
        return False


def process_session(session_id: int, progress_callback=None):
    """
    Process a session: fetch data, chunk, embed, and analyze.
    """
    session = get_session(session_id)
    if not session:
        return False, "Session not found"

    update_session_status(session_id, "processing")

    total_steps = 4
    current_step = 0

    try:
        # Step 1: Fetch App Store/Play Store reviews
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps, "Fetching app reviews...")

        app_ids = session.get('app_ids') or {}
        if isinstance(app_ids, str):
            app_ids = json.loads(app_ids)

        all_reviews = []

        # Process App Store IDs
        for app_id in app_ids.get('app_store', []):
            if app_id:
                reviews = fetch_app_store_reviews(
                    app_id,
                    days_back=session.get('time_window_days', 90)
                )
                for review in reviews:
                    content = f"{review.get('title', '')} {review.get('content', '')}"
                    doc_id = save_document(
                        session_id=session_id,
                        source_type='app_store_review',
                        content=content,
                        source_url=review.get('source_url'),
                        source_name=f"App Store Review - {app_id}",
                        metadata={
                            'rating': review.get('rating'),
                            'date': review.get('date'),
                            'author': review.get('author')
                        }
                    )
                    all_reviews.append({
                        'doc_id': doc_id,
                        'content': content,
                        'source_url': review.get('source_url')
                    })

        # Process Play Store IDs (note: limited without API)
        for package_name in app_ids.get('play_store', []):
            if package_name:
                reviews = fetch_play_store_reviews_rss(
                    package_name,
                    days_back=session.get('time_window_days', 90)
                )
                for review in reviews:
                    content = f"{review.get('title', '')} {review.get('content', '')}"
                    doc_id = save_document(
                        session_id=session_id,
                        source_type='play_store_note',
                        content=content,
                        source_url=review.get('source_url'),
                        source_name=f"Play Store - {package_name}",
                        metadata={}
                    )

        # Step 2: Fetch changelogs/blogs
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps, "Fetching changelogs and blogs...")

        changelog_urls = session.get('changelog_urls') or []
        for url in changelog_urls:
            if url:
                content_items = fetch_changelog_content(
                    url,
                    days_back=session.get('time_window_days', 90)
                )
                for item in content_items:
                    content = f"{item.get('title', '')} {item.get('content', '')}"
                    doc_id = save_document(
                        session_id=session_id,
                        source_type='changelog',
                        content=content,
                        source_url=item.get('url', url),
                        source_name=item.get('title', url),
                        metadata={'date': item.get('date')}
                    )
                    all_reviews.append({
                        'doc_id': doc_id,
                        'content': content,
                        'source_url': item.get('url', url)
                    })

        # Step 3: Chunk and embed
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps, "Processing and embedding text...")

        documents = get_documents(session_id)
        all_chunks = []
        chunk_metadata = []

        for doc in documents:
            chunks = chunk_text(doc['content'], chunk_size=800, overlap=100)
            for chunk in chunks:
                all_chunks.append(chunk['text'])
                chunk_metadata.append({
                    'doc_id': doc['id'],
                    'chunk_index': chunk['index'],
                    'source_url': doc.get('source_url')
                })

        # Get embeddings in batches
        if all_chunks:
            embeddings = get_embeddings_batch(all_chunks, use_cache=True)

            # Save vectors
            for i, (text, embedding, meta) in enumerate(zip(all_chunks, embeddings, chunk_metadata)):
                chunk_data = [{
                    'text': text,
                    'index': meta['chunk_index'],
                    'embedding': embedding,
                    'metadata': {'source_url': meta['source_url']}
                }]
                save_vectors(session_id, meta['doc_id'], chunk_data)

        # Step 4: Compute pain point clusters
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps, "Analyzing pain points...")

        clear_insights(session_id)

        # Extract texts and URLs for pain point analysis
        texts = [doc['content'] for doc in documents]
        urls = [doc.get('source_url') for doc in documents]

        pain_points = extract_pain_points(texts, urls, min_frequency=2)

        for pp in pain_points:
            save_insight(
                session_id=session_id,
                label=pp['label'],
                frequency=pp['frequency'],
                sentiment_score=pp['sentiment_score'],
                evidence_urls=pp['evidence_urls'],
                sample_texts=pp['sample_texts']
            )

        update_session_status(session_id, "completed")
        return True, f"Successfully processed {len(documents)} documents and found {len(pain_points)} pain points"

    except Exception as e:
        update_session_status(session_id, "error")
        return False, str(e)


# Main app
st.title("🔍 AI Product Research Agent")
st.markdown("""
Automate product research and competitive analysis using AI.
Analyze app reviews, changelogs, and blogs to identify pain points and insights.
""")

# Initialize
if not init_app():
    st.stop()

# Sidebar - Session list
with st.sidebar:
    st.header("📋 Sessions")

    sessions = get_all_sessions()
    if sessions:
        for session in sessions:
            status_emoji = {
                'created': '🆕',
                'processing': '⏳',
                'completed': '✅',
                'error': '❌'
            }.get(session['status'], '❓')

            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"{status_emoji} {session['name']}", key=f"session_{session['id']}", use_container_width=True):
                    st.session_state['selected_session'] = session['id']
            with col2:
                if st.button("🗑️", key=f"delete_{session['id']}", help="Delete session"):
                    delete_session(session['id'])
                    st.rerun()
    else:
        st.info("No sessions yet. Create one below!")

    st.divider()
    st.markdown("[📊 Explore Insights →](./Explore)")

# Main content
tab1, tab2 = st.tabs(["➕ Create Session", "📖 Session Details"])

with tab1:
    st.header("Create New Research Session")

    with st.form("create_session_form"):
        # Session name
        session_name = st.text_input(
            "Session Name",
            placeholder="e.g., Note-taking Apps Analysis Q1 2024"
        )

        # Research goal
        research_goal = st.text_area(
            "Research Goal",
            placeholder="e.g., Identify top pain points in onboarding for note-taking apps",
            help="Describe what you want to learn from this research"
        )

        # Competitors
        competitors = st.text_area(
            "Competitors (one per line)",
            placeholder="Notion\nEvernote\nObsidian\nRoam Research",
            help="List the competitor products you want to analyze"
        )

        # App IDs section
        st.subheader("📱 App Store Data (Optional)")
        show_app_ids = st.checkbox("Add App Store / Play Store IDs", value=False)

        app_store_ids = ""
        play_store_ids = ""

        if show_app_ids:
            st.info("""
            **How to find App IDs:**

            **App Store (iOS):**
            - Go to the app page on apps.apple.com
            - The ID is the number after `/id` in the URL
            - Example: `https://apps.apple.com/us/app/notion/id1232780281` → ID is `1232780281`
            - You can also paste the full URL

            **Play Store (Android):**
            - Go to the app page on play.google.com
            - The ID is the `id` parameter in the URL
            - Example: `https://play.google.com/store/apps/details?id=notion.id` → ID is `notion.id`
            - You can also paste the full URL or just the package name

            *Note: Play Store reviews require the official API for full access. We'll fetch what's publicly available.*
            """)

            col1, col2 = st.columns(2)
            with col1:
                app_store_ids = st.text_area(
                    "App Store IDs/URLs (one per line)",
                    placeholder="1232780281\nhttps://apps.apple.com/us/app/evernote/id281796108",
                    help="Enter App Store numeric IDs or full URLs"
                )
            with col2:
                play_store_ids = st.text_area(
                    "Play Store Package Names/URLs (one per line)",
                    placeholder="notion.id\ncom.evernote",
                    help="Enter Play Store package names or full URLs"
                )

        # Changelog URLs section
        st.subheader("📝 Changelog/Blog URLs (Optional)")
        show_changelog = st.checkbox("Add Changelog or Blog URLs", value=False)

        changelog_urls_input = ""
        if show_changelog:
            st.info("""
            **Supported sources:**
            - RSS/Atom feeds (automatically detected)
            - Blog pages with changelog posts
            - Release notes pages

            *We respect robots.txt and terms of service.*
            """)
            changelog_urls_input = st.text_area(
                "Changelog/Blog URLs (one per line)",
                placeholder="https://www.notion.so/releases\nhttps://evernote.com/blog",
                help="Enter URLs to changelog pages or blogs"
            )

        # Time window
        time_window = st.slider(
            "Time Window (days)",
            min_value=7,
            max_value=365,
            value=90,
            help="Only fetch data from the last N days"
        )

        # Submit button
        submitted = st.form_submit_button("🚀 Create Session", use_container_width=True)

        if submitted:
            if not session_name or not research_goal:
                st.error("Please provide a session name and research goal.")
            else:
                # Parse inputs
                competitors_list = [c.strip() for c in competitors.split('\n') if c.strip()]

                # Parse app IDs
                app_store_list = []
                play_store_list = []

                if app_store_ids:
                    for line in app_store_ids.split('\n'):
                        line = line.strip()
                        if line:
                            app_id, store_type = parse_app_id(line)
                            app_store_list.append(app_id)

                if play_store_ids:
                    for line in play_store_ids.split('\n'):
                        line = line.strip()
                        if line:
                            app_id, store_type = parse_app_id(line)
                            play_store_list.append(app_id)

                app_ids_dict = {
                    'app_store': app_store_list,
                    'play_store': play_store_list
                }

                changelog_urls_list = [u.strip() for u in changelog_urls_input.split('\n') if u.strip()]

                # Create session
                try:
                    session_id = create_session(
                        name=session_name,
                        research_goal=research_goal,
                        competitors=competitors_list,
                        app_ids=app_ids_dict,
                        changelog_urls=changelog_urls_list,
                        time_window_days=time_window
                    )

                    st.success(f"✅ Session created! ID: {session_id}")
                    st.session_state['selected_session'] = session_id

                    # Process the session
                    st.info("Starting data ingestion...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        status_text.text(message)

                    success, message = process_session(session_id, update_progress)

                    if success:
                        st.success(f"✅ {message}")
                        st.info("Head to the **Explore** page to view insights!")
                    else:
                        st.error(f"❌ Error: {message}")

                    st.rerun()

                except Exception as e:
                    st.error(f"Error creating session: {str(e)}")

with tab2:
    if 'selected_session' in st.session_state:
        session = get_session(st.session_state['selected_session'])
        if session:
            st.header(f"📖 {session['name']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", session['status'].title())
            with col2:
                st.metric("Time Window", f"{session['time_window_days']} days")
            with col3:
                created = session['created_at']
                if isinstance(created, datetime):
                    st.metric("Created", created.strftime("%Y-%m-%d"))
                else:
                    st.metric("Created", str(created)[:10])

            st.subheader("Research Goal")
            st.write(session['research_goal'])

            if session.get('competitors'):
                st.subheader("Competitors")
                st.write(", ".join(session['competitors']))

            app_ids = session.get('app_ids') or {}
            if isinstance(app_ids, str):
                app_ids = json.loads(app_ids)

            if app_ids.get('app_store') or app_ids.get('play_store'):
                st.subheader("App IDs")
                if app_ids.get('app_store'):
                    st.write(f"**App Store:** {', '.join(app_ids['app_store'])}")
                if app_ids.get('play_store'):
                    st.write(f"**Play Store:** {', '.join(app_ids['play_store'])}")

            if session.get('changelog_urls'):
                st.subheader("Changelog URLs")
                for url in session['changelog_urls']:
                    st.write(f"- {url}")

            # Documents info
            documents = get_documents(session['id'])
            if documents:
                st.subheader("📄 Fetched Documents")
                st.metric("Total Documents", len(documents))

                doc_types = {}
                for doc in documents:
                    doc_type = doc.get('source_type', 'unknown')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

                for doc_type, count in doc_types.items():
                    st.write(f"- {doc_type}: {count}")

            # Action buttons
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Re-process Session", use_container_width=True):
                    st.info("Re-processing session...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        status_text.text(message)

                    success, message = process_session(session['id'], update_progress)
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ Error: {message}")
                    st.rerun()

            with col2:
                if st.button("📊 View Insights", use_container_width=True):
                    st.switch_page("pages/Explore.py")

    else:
        st.info("Select a session from the sidebar or create a new one.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
AI Product Research Agent | Powered by OpenAI & Supabase<br>
⚠️ Respects robots.txt and Terms of Service | For educational/research purposes
</div>
""", unsafe_allow_html=True)
