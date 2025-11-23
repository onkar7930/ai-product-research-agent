"""
Supabase Database utilities for AI Product Research Agent.
Handles all database operations including sessions, documents, vectors, and insights.
"""

import os
import json
import hashlib
from datetime import datetime
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
    """Initialize database tables if they don't exist."""
    conn = get_connection()
    cur = conn.cursor()

    # Sessions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            research_goal TEXT NOT NULL,
            competitors TEXT[],
            app_ids JSONB,
            changelog_urls TEXT[],
            time_window_days INTEGER DEFAULT 90,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(50) DEFAULT 'created'
        )
    """)

    # Documents table - stores fetched content
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
            source_type VARCHAR(50) NOT NULL,
            source_url TEXT,
            source_name VARCHAR(255),
            content TEXT NOT NULL,
            metadata JSONB,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Vectors table - stores chunked text with embeddings
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            id SERIAL PRIMARY KEY,
            session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            chunk_text TEXT NOT NULL,
            chunk_index INTEGER,
            embedding JSONB,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insights table - stores analyzed pain points
    cur.execute("""
        CREATE TABLE IF NOT EXISTS insights (
            id SERIAL PRIMARY KEY,
            session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
            label VARCHAR(255) NOT NULL,
            frequency INTEGER DEFAULT 1,
            sentiment_score FLOAT,
            evidence_count INTEGER DEFAULT 0,
            evidence_urls TEXT[],
            sample_texts TEXT[],
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Cache table - for embedding cache to minimize API costs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            id SERIAL PRIMARY KEY,
            cache_key VARCHAR(64) UNIQUE NOT NULL,
            cache_type VARCHAR(50) NOT NULL,
            data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        )
    """)

    # Create indexes for better performance
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_vectors_session ON vectors(session_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_insights_session ON insights(session_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cache_key ON cache(cache_key)")

    conn.commit()
    cur.close()
    conn.close()


# Session operations
def create_session(name: str, research_goal: str, competitors: List[str],
                   app_ids: Dict[str, List[str]], changelog_urls: List[str],
                   time_window_days: int = 90) -> int:
    """Create a new research session."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO sessions (name, research_goal, competitors, app_ids, changelog_urls, time_window_days)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (name, research_goal, competitors, json.dumps(app_ids), changelog_urls, time_window_days))

    session_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    return session_id


def get_session(session_id: int) -> Optional[Dict]:
    """Get a session by ID."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
    session = cur.fetchone()

    cur.close()
    conn.close()

    return dict(session) if session else None


def get_all_sessions() -> List[Dict]:
    """Get all sessions."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT * FROM sessions ORDER BY created_at DESC")
    sessions = cur.fetchall()

    cur.close()
    conn.close()

    return [dict(s) for s in sessions]


def update_session_status(session_id: int, status: str):
    """Update session status."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("UPDATE sessions SET status = %s WHERE id = %s", (status, session_id))

    conn.commit()
    cur.close()
    conn.close()


# Document operations
def save_document(session_id: int, source_type: str, content: str,
                  source_url: str = None, source_name: str = None,
                  metadata: Dict = None) -> int:
    """Save a fetched document."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO documents (session_id, source_type, source_url, source_name, content, metadata)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (session_id, source_type, source_url, source_name, content,
          json.dumps(metadata) if metadata else None))

    doc_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    return doc_id


def get_documents(session_id: int) -> List[Dict]:
    """Get all documents for a session."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT * FROM documents WHERE session_id = %s", (session_id,))
    docs = cur.fetchall()

    cur.close()
    conn.close()

    return [dict(d) for d in docs]


# Vector operations
def save_vectors(session_id: int, document_id: int, chunks: List[Dict]):
    """Save multiple vector chunks."""
    if not chunks:
        return

    conn = get_connection()
    cur = conn.cursor()

    data = [(session_id, document_id, c['text'], c['index'],
             json.dumps(c['embedding']), json.dumps(c.get('metadata', {})))
            for c in chunks]

    execute_values(cur, """
        INSERT INTO vectors (session_id, document_id, chunk_text, chunk_index, embedding, metadata)
        VALUES %s
    """, data)

    conn.commit()
    cur.close()
    conn.close()


def get_vectors(session_id: int) -> List[Dict]:
    """Get all vectors for a session."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT v.*, d.source_url, d.source_name, d.source_type
        FROM vectors v
        JOIN documents d ON v.document_id = d.id
        WHERE v.session_id = %s
    """, (session_id,))
    vectors = cur.fetchall()

    cur.close()
    conn.close()

    return [dict(v) for v in vectors]


# Insight operations
def save_insight(session_id: int, label: str, frequency: int, sentiment_score: float,
                 evidence_urls: List[str], sample_texts: List[str], metadata: Dict = None) -> int:
    """Save an insight/pain point."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO insights (session_id, label, frequency, sentiment_score,
                              evidence_count, evidence_urls, sample_texts, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (session_id, label, frequency, sentiment_score, len(evidence_urls),
          evidence_urls, sample_texts, json.dumps(metadata) if metadata else None))

    insight_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    return insight_id


def get_insights(session_id: int) -> List[Dict]:
    """Get all insights for a session."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT * FROM insights
        WHERE session_id = %s
        ORDER BY frequency DESC, sentiment_score ASC
    """, (session_id,))
    insights = cur.fetchall()

    cur.close()
    conn.close()

    return [dict(i) for i in insights]


def clear_insights(session_id: int):
    """Clear existing insights for a session (for re-analysis)."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DELETE FROM insights WHERE session_id = %s", (session_id,))

    conn.commit()
    cur.close()
    conn.close()


# Cache operations
def get_cached_embedding(text: str) -> Optional[List[float]]:
    """Get cached embedding for text."""
    cache_key = hashlib.sha256(text.encode()).hexdigest()

    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT data FROM cache
        WHERE cache_key = %s AND cache_type = 'embedding'
        AND (expires_at IS NULL OR expires_at > NOW())
    """, (cache_key,))

    result = cur.fetchone()
    cur.close()
    conn.close()

    if result:
        return result['data']
    return None


def save_cached_embedding(text: str, embedding: List[float]):
    """Save embedding to cache."""
    cache_key = hashlib.sha256(text.encode()).hexdigest()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO cache (cache_key, cache_type, data)
        VALUES (%s, 'embedding', %s)
        ON CONFLICT (cache_key) DO UPDATE SET data = EXCLUDED.data
    """, (cache_key, json.dumps(embedding)))

    conn.commit()
    cur.close()
    conn.close()


def delete_session(session_id: int):
    """Delete a session and all related data."""
    conn = get_connection()
    cur = conn.cursor()

    # Cascading delete will handle documents, vectors, insights
    cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))

    conn.commit()
    cur.close()
    conn.close()
