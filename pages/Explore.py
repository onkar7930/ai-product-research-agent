"""
AI Product Research Agent - Explore Page
View insights, ask questions, and export data from research sessions.
"""

import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime

st.set_page_config(
    page_title="Explore Insights | AI Product Research Agent",
    page_icon="📊",
    layout="wide"
)

from utils.database import (
    init_database, get_all_sessions, get_session,
    get_insights, get_vectors, get_documents
)
from utils.rag import answer_question, summarize_insights
from utils.embeddings import get_embedding


def init_page():
    """Initialize the page."""
    try:
        init_database()
        return True
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return False


# Initialize
if not init_page():
    st.stop()

st.title("📊 Explore Insights")

# Session selector
sessions = get_all_sessions()

if not sessions:
    st.warning("No sessions found. Please create a session first.")
    if st.button("← Go to Home"):
        st.switch_page("Home.py")
    st.stop()

# Filter to completed sessions
completed_sessions = [s for s in sessions if s['status'] == 'completed']

if not completed_sessions:
    st.warning("No completed sessions found. Please wait for processing to complete or create a new session.")
    if st.button("← Go to Home"):
        st.switch_page("Home.py")
    st.stop()

# Session selector in sidebar
with st.sidebar:
    st.header("🎯 Select Session")

    session_options = {s['id']: f"{s['name']} ({s['created_at'].strftime('%Y-%m-%d') if isinstance(s['created_at'], datetime) else str(s['created_at'])[:10]})" for s in completed_sessions}

    selected_session_id = st.selectbox(
        "Choose a session",
        options=list(session_options.keys()),
        format_func=lambda x: session_options[x]
    )

    if selected_session_id:
        session = get_session(selected_session_id)
        if session:
            st.divider()
            st.subheader("📋 Session Info")
            st.write(f"**Goal:** {session['research_goal'][:100]}...")
            if session.get('competitors'):
                st.write(f"**Competitors:** {', '.join(session['competitors'][:3])}...")

            docs = get_documents(selected_session_id)
            st.metric("Documents", len(docs))

    st.divider()
    st.markdown("[← Back to Home](./)")

# Main content
if selected_session_id:
    session = get_session(selected_session_id)
    insights = get_insights(selected_session_id)
    vectors = get_vectors(selected_session_id)

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Pain Points", "💬 Ask Questions", "📈 Summary", "📥 Export"])

    # Tab 1: Pain Points Table
    with tab1:
        st.header("Pain Point Clusters")

        if not insights:
            st.info("No pain points found. The session may need more data or re-processing.")
        else:
            # Sentiment filter
            col1, col2 = st.columns([2, 1])
            with col1:
                sentiment_filter = st.select_slider(
                    "Filter by Sentiment",
                    options=["All", "Negative Only", "Neutral Only", "Positive Only"],
                    value="All"
                )
            with col2:
                min_frequency = st.number_input("Min Frequency", min_value=1, value=2)

            # Filter insights
            filtered_insights = insights

            if sentiment_filter == "Negative Only":
                filtered_insights = [i for i in insights if i['sentiment_score'] < -0.2]
            elif sentiment_filter == "Neutral Only":
                filtered_insights = [i for i in insights if -0.2 <= i['sentiment_score'] <= 0.2]
            elif sentiment_filter == "Positive Only":
                filtered_insights = [i for i in insights if i['sentiment_score'] > 0.2]

            filtered_insights = [i for i in filtered_insights if i['frequency'] >= min_frequency]

            # Create DataFrame
            if filtered_insights:
                df = pd.DataFrame([
                    {
                        'Pain Point': i['label'],
                        'Frequency': i['frequency'],
                        'Sentiment': round(i['sentiment_score'], 2),
                        'Evidence Count': i['evidence_count'],
                        'Sentiment Label': 'Negative' if i['sentiment_score'] < -0.2 else 'Positive' if i['sentiment_score'] > 0.2 else 'Neutral'
                    }
                    for i in filtered_insights
                ])

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Pain Points", len(filtered_insights))
                with col2:
                    negative_count = len([i for i in filtered_insights if i['sentiment_score'] < -0.2])
                    st.metric("Negative Issues", negative_count)
                with col3:
                    avg_frequency = sum(i['frequency'] for i in filtered_insights) / len(filtered_insights)
                    st.metric("Avg Frequency", f"{avg_frequency:.1f}")
                with col4:
                    total_evidence = sum(i['evidence_count'] for i in filtered_insights)
                    st.metric("Total Evidence", total_evidence)

                st.divider()

                # Display table with color coding
                st.dataframe(
                    df.style.background_gradient(
                        subset=['Frequency'],
                        cmap='Reds'
                    ).background_gradient(
                        subset=['Sentiment'],
                        cmap='RdYlGn',
                        vmin=-1,
                        vmax=1
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                # Expandable details
                st.subheader("📝 Details")
                for i, insight in enumerate(filtered_insights[:20]):
                    with st.expander(f"**{insight['label']}** (freq: {insight['frequency']}, sentiment: {insight['sentiment_score']:.2f})"):
                        if insight.get('sample_texts'):
                            st.write("**Sample Evidence:**")
                            for sample in insight['sample_texts'][:3]:
                                st.markdown(f"> {sample}")

                        if insight.get('evidence_urls'):
                            st.write("**Source URLs:**")
                            for url in insight['evidence_urls'][:5]:
                                st.markdown(f"- [{url[:50]}...]({url})" if len(url) > 50 else f"- [{url}]({url})")
            else:
                st.info("No pain points match the current filters.")

    # Tab 2: Ask Questions (RAG)
    with tab2:
        st.header("Ask Questions About Your Research")
        st.markdown("""
        Ask natural language questions about the research data.
        The AI will search through all collected documents and provide answers with citations.
        """)

        # Example questions
        with st.expander("💡 Example Questions"):
            st.markdown("""
            - What are the most common complaints about the onboarding process?
            - What features do users most frequently request?
            - How do users feel about the pricing?
            - What bugs or technical issues are mentioned most often?
            - What do users like most about the competitor products?
            - Are there any sync or performance issues mentioned?
            """)

        # Question input
        question = st.text_input(
            "Your Question",
            placeholder="e.g., What are the main pain points users have with syncing?",
            key="rag_question"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            num_sources = st.number_input("Sources to use", min_value=3, max_value=10, value=5)

        if st.button("🔍 Get Answer", use_container_width=True):
            if question:
                with st.spinner("Searching and generating answer..."):
                    result = answer_question(
                        question=question,
                        vectors=vectors,
                        research_goal=session.get('research_goal', ''),
                        top_k=num_sources
                    )

                    # Display answer
                    st.subheader("Answer")

                    # Confidence indicator
                    confidence_colors = {
                        'high': '🟢',
                        'medium': '🟡',
                        'low': '🟠',
                        'none': '🔴',
                        'error': '❌'
                    }
                    confidence = result.get('confidence', 'unknown')
                    st.write(f"{confidence_colors.get(confidence, '❓')} Confidence: {confidence.title()} | Sources used: {result.get('num_sources', 0)}")

                    st.markdown(result['answer'])

                    # Citations
                    if result.get('citations'):
                        st.subheader("📚 Sources")
                        for citation in result['citations']:
                            relevance_pct = citation.get('relevance', 0) * 100
                            st.markdown(f"""
                            **[Source {citation['index']}]** {citation.get('source_name', 'Unknown')} ({citation.get('source_type', 'document')})
                            - URL: [{citation.get('url', 'N/A')[:60]}...]({citation.get('url', '#')})
                            - Relevance: {relevance_pct:.0f}%
                            """)
            else:
                st.warning("Please enter a question.")

        # Recent questions (stored in session state)
        if 'recent_questions' not in st.session_state:
            st.session_state.recent_questions = []

    # Tab 3: Summary
    with tab3:
        st.header("Executive Summary")

        if st.button("🔄 Generate Summary", use_container_width=True):
            with st.spinner("Generating executive summary..."):
                summary = summarize_insights(
                    insights=insights,
                    research_goal=session.get('research_goal', '')
                )
                st.session_state['summary'] = summary

        if 'summary' in st.session_state:
            st.markdown(st.session_state['summary'])

        st.divider()

        # Key metrics
        st.subheader("📊 Key Metrics")

        if insights:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Pain Points", len(insights))
                negative_pps = [i for i in insights if i['sentiment_score'] < -0.2]
                st.metric("Negative Issues", len(negative_pps))

            with col2:
                if insights:
                    top_pain_point = max(insights, key=lambda x: x['frequency'])
                    st.metric("Top Pain Point", top_pain_point['label'][:30])
                    st.metric("Highest Frequency", top_pain_point['frequency'])

            with col3:
                total_evidence = sum(i['evidence_count'] for i in insights)
                st.metric("Total Evidence", total_evidence)
                avg_sentiment = sum(i['sentiment_score'] for i in insights) / len(insights)
                st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")

            # Top 5 pain points chart
            st.subheader("📈 Top Pain Points by Frequency")
            top_5 = sorted(insights, key=lambda x: x['frequency'], reverse=True)[:5]
            chart_data = pd.DataFrame({
                'Pain Point': [i['label'] for i in top_5],
                'Frequency': [i['frequency'] for i in top_5]
            })
            st.bar_chart(chart_data.set_index('Pain Point'))

    # Tab 4: Export
    with tab4:
        st.header("Export Data")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📥 Export Pain Points")

            if insights:
                # Create CSV
                df = pd.DataFrame([
                    {
                        'Label': i['label'],
                        'Frequency': i['frequency'],
                        'Sentiment Score': i['sentiment_score'],
                        'Evidence Count': i['evidence_count'],
                        'Evidence URLs': '; '.join(i.get('evidence_urls', [])[:5]),
                        'Sample Text': (i.get('sample_texts', [''])[0])[:200] if i.get('sample_texts') else ''
                    }
                    for i in insights
                ])

                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="📄 Download Pain Points CSV",
                    data=csv_data,
                    file_name=f"pain_points_{session['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.info(f"Contains {len(insights)} pain points")
            else:
                st.warning("No pain points to export.")

        with col2:
            st.subheader("📥 Export Documents")

            docs = get_documents(selected_session_id)

            if docs:
                # Create documents CSV
                docs_df = pd.DataFrame([
                    {
                        'Source Type': d.get('source_type', ''),
                        'Source Name': d.get('source_name', ''),
                        'Source URL': d.get('source_url', ''),
                        'Content Preview': d.get('content', '')[:300],
                        'Fetched At': str(d.get('fetched_at', ''))
                    }
                    for d in docs
                ])

                docs_csv_buffer = io.StringIO()
                docs_df.to_csv(docs_csv_buffer, index=False)
                docs_csv_data = docs_csv_buffer.getvalue()

                st.download_button(
                    label="📄 Download Documents CSV",
                    data=docs_csv_data,
                    file_name=f"documents_{session['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.info(f"Contains {len(docs)} documents")
            else:
                st.warning("No documents to export.")

        st.divider()

        # Full export (JSON)
        st.subheader("📦 Full Session Export (JSON)")

        export_data = {
            'session': {
                'id': session['id'],
                'name': session['name'],
                'research_goal': session['research_goal'],
                'competitors': session.get('competitors', []),
                'time_window_days': session.get('time_window_days'),
                'created_at': str(session.get('created_at', '')),
                'status': session.get('status', '')
            },
            'insights': [
                {
                    'label': i['label'],
                    'frequency': i['frequency'],
                    'sentiment_score': i['sentiment_score'],
                    'evidence_count': i['evidence_count'],
                    'evidence_urls': i.get('evidence_urls', []),
                    'sample_texts': i.get('sample_texts', [])
                }
                for i in insights
            ],
            'documents_count': len(docs) if docs else 0,
            'vectors_count': len(vectors) if vectors else 0,
            'exported_at': datetime.now().isoformat()
        }

        json_data = json.dumps(export_data, indent=2)

        st.download_button(
            label="📦 Download Full Export (JSON)",
            data=json_data,
            file_name=f"full_export_{session['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )

else:
    st.warning("Please select a session from the sidebar.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
AI Product Research Agent | Powered by OpenAI & Supabase
</div>
""", unsafe_allow_html=True)
