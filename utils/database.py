"""
Supabase Database utilities for AI Product Research Agent.
Handles all database operations including sessions, documents, vectors, and insights.
Adapted to work with existing Supabase schema using UUID primary keys.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import streamlit as st


def get_connection():
    """Get database connection using Supabase credentials."""
    return psycopg2.connect(
        host=os.getenv("SUPABASE_HOST"),
        database=os.getenv("SUPABASE_DATABASE", "postgres"),
        user=os.getenv("SUPABASE_USER", "postgres"),
        password=os.getenv("SUPABASE_PASSWORD"),
        port=os.getenv("SUPABASE_PORT", "5432"),
        sslmode="require"
    )


def init_database():
    """Verify database connection and tables exist."""
    conn = get_connection()
    cur = conn.cursor()

    # Just verify we can connect and tables exist
    cur.execute("SELECT 1 FROM sessions LIMIT 1")

    cur.close()
    conn.close()


# Session operations
def create_session(name: str, research_goal: str, competitors: List[str],
                   app_ids: Dict[str, List[str]], changelog_urls: List[str],
                   time_window_days: int = 90) -> str:
    """Create a new research session. Returns UUID as string."""
    conn = get_connection()
    cur = conn.cursor()

    # Combine app_ids and changelog_urls into sources
    sources = changelog_urls.copy()
    for store, ids in app_ids.items():
        for app_id in ids:
            if store == 'app_store':
                sources.append(f"appstore:{app_id}")
            else:
                sources.append(f"playstore:{app_id}")

    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=time_window_days)

    cur.execute("""
        INSERT INTO sessions (title, goal, competitors, sources, start_date, end_date)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (name, research_goal, competitors, sources, start_date, end_date))

    session_id = str(cur.fetchone()[0])
    conn.commit()
    cur.close()
    conn.close()

    return session_id


def get_session(session_id: str) -> Optional[Dict]:
    """Get a session by ID."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
    session = cur.fetchone()

    cur.close()
    conn.close()

    if session:
        result = dict(session)
        # Map to expected field names
        result['name'] = result.get('title', '')
        result['research_goal'] = result.get('goal', '')
        result['time_window_days'] = 90  # Default
        if result.get('start_date') and result.get('end_date'):
            delta = result['end_date'] - result['start_date']
            result['time_window_days'] = delta.days
        result['status'] = result.get('status', 'created')

        # Parse sources back to app_ids
        sources = result.get('sources', []) or []
        app_ids = {'app_store': [], 'play_store': []}
        changelog_urls = []
        for src in sources:
            if src.startswith('appstore:'):
                app_ids['app_store'].append(src.replace('appstore:', ''))
            elif src.startswith('playstore:'):
                app_ids['play_store'].append(src.replace('playstore:', ''))
            else:
                changelog_urls.append(src)
        result['app_ids'] = app_ids
        result['changelog_urls'] = changelog_urls

        return result
    return None


def get_all_sessions() -> List[Dict]:
    """Get all sessions."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT * FROM sessions ORDER BY created_at DESC")
    sessions = cur.fetchall()

    cur.close()
    conn.close()

    results = []
    for session in sessions:
        result = dict(session)
        result['name'] = result.get('title', '')
        result['research_goal'] = result.get('goal', '')
        result['status'] = result.get('status', 'completed')  # Assume completed if no status
        results.append(result)

    return results


def update_session_status(session_id: str, status: str):
    """Update session status. Note: Your schema doesn't have status column, so we'll skip this."""
    # Your schema doesn't have a status column - we'll just pass
    pass


# Document operations
def save_document(session_id: str, source_type: str, content: str,
                  source_url: str = None, source_name: str = None,
                  metadata: Dict = None) -> str:
    """Save a fetched document. Returns UUID as string."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO documents (session_id, url, source, text, meta)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
    """, (session_id, source_url, source_type, content,
          json.dumps(metadata) if metadata else None))

    doc_id = str(cur.fetchone()[0])
    conn.commit()
    cur.close()
    conn.close()

    return doc_id


def get_documents(session_id: str) -> List[Dict]:
    """Get all documents for a session."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT * FROM documents WHERE session_id = %s", (session_id,))
    docs = cur.fetchall()

    cur.close()
    conn.close()

    results = []
    for d in docs:
        result = dict(d)
        # Map to expected field names
        result['content'] = result.get('text', '')
        result['source_url'] = result.get('url', '')
        result['source_type'] = result.get('source', '')
        result['source_name'] = result.get('source', '')
        results.append(result)

    return results


# Vector operations
def save_vectors(session_id: str, document_id: str, chunks: List[Dict]):
    """Save multiple vector chunks."""
    if not chunks:
        return

    conn = get_connection()
    cur = conn.cursor()

    for c in chunks:
        cur.execute("""
            INSERT INTO vectors (doc_id, chunk_id, embedding, text)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (doc_id, chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding, text = EXCLUDED.text
        """, (document_id, c['index'], json.dumps(c['embedding']), c['text']))

    conn.commit()
    cur.close()
    conn.close()


def get_vectors(session_id: str) -> List[Dict]:
    """Get all vectors for a session."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT v.*, d.url, d.source
        FROM vectors v
        JOIN documents d ON v.doc_id = d.id
        WHERE d.session_id = %s
    """, (session_id,))
    vectors = cur.fetchall()

    cur.close()
    conn.close()

    results = []
    for v in vectors:
        result = dict(v)
        result['chunk_text'] = result.get('text', '')
        result['source_url'] = result.get('url', '')
        result['source_type'] = result.get('source', '')
        result['source_name'] = result.get('source', '')
        results.append(result)

    return results


# Insight operations
def save_insight(session_id: str, label: str, frequency: int, sentiment_score: float,
                 evidence_urls: List[str], sample_texts: List[str], metadata: Dict = None) -> str:
    """Save an insight/pain point. Returns UUID as string."""
    conn = get_connection()
    cur = conn.cursor()

    payload = {
        'label': label,
        'frequency': frequency,
        'sentiment_score': sentiment_score,
        'sample_texts': sample_texts,
        'evidence_count': len(evidence_urls)
    }

    cur.execute("""
        INSERT INTO insights (session_id, type, payload, evidence_urls)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """, (session_id, 'pain_point', json.dumps(payload), evidence_urls))

    insight_id = str(cur.fetchone()[0])
    conn.commit()
    cur.close()
    conn.close()

    return insight_id


def get_insights(session_id: str) -> List[Dict]:
    """Get all insights for a session."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT * FROM insights
        WHERE session_id = %s
        ORDER BY (payload->>'frequency')::int DESC NULLS LAST
    """, (session_id,))
    insights = cur.fetchall()

    cur.close()
    conn.close()

    results = []
    for i in insights:
        result = dict(i)
        payload = result.get('payload', {})
        if isinstance(payload, str):
            payload = json.loads(payload)

        result['label'] = payload.get('label', '')
        result['frequency'] = payload.get('frequency', 0)
        result['sentiment_score'] = payload.get('sentiment_score', 0)
        result['sample_texts'] = payload.get('sample_texts', [])
        result['evidence_count'] = payload.get('evidence_count', 0)
        result['evidence_urls'] = result.get('evidence_urls', [])
        results.append(result)

    return results


def clear_insights(session_id: str):
    """Clear existing insights for a session (for re-analysis)."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DELETE FROM insights WHERE session_id = %s", (session_id,))

    conn.commit()
    cur.close()
    conn.close()


# Cache operations (using vectors_cache table)
def get_cached_embedding(text: str) -> Optional[List[float]]:
    """Get cached embedding for text."""
    cache_key = hashlib.sha256(text.encode()).hexdigest()

    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT embedding FROM vectors_cache
        WHERE hash = %s
    """, (cache_key,))

    result = cur.fetchone()
    cur.close()
    conn.close()

    if result:
        embedding = result['embedding']
        if isinstance(embedding, str):
            return json.loads(embedding)
        return embedding
    return None


def save_cached_embedding(text: str, embedding: List[float]):
    """Save embedding to cache."""
    cache_key = hashlib.sha256(text.encode()).hexdigest()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO vectors_cache (hash, embedding)
        VALUES (%s, %s)
        ON CONFLICT (hash) DO UPDATE SET embedding = EXCLUDED.embedding
    """, (cache_key, json.dumps(embedding)))

    conn.commit()
    cur.close()
    conn.close()


def delete_session(session_id: str):
    """Delete a session and all related data."""
    conn = get_connection()
    cur = conn.cursor()

    # Cascading delete will handle documents, vectors, insights
    cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))

    conn.commit()
    cur.close()
    conn.close()
