"""
RAG (Retrieval Augmented Generation) utilities.
Handles question answering over the session corpus.
"""

import os
import requests
from typing import List, Dict
from .embeddings import get_embedding, find_similar_chunks


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"  # Cost-effective model for RAG


def generate_answer(question: str, context_chunks: List[Dict],
                    research_goal: str = "") -> Dict:
    """
    Generate an answer to a question using retrieved context chunks.

    Args:
        question: The user's question
        context_chunks: List of relevant chunks with text and metadata
        research_goal: The original research goal for context

    Returns:
        Dict with 'answer', 'citations', and 'confidence'
    """
    # Build context from chunks
    context_parts = []
    citations = []

    for i, chunk in enumerate(context_chunks):
        source_info = f"[Source {i+1}]"
        if chunk.get('source_name'):
            source_info += f" {chunk['source_name']}"
        if chunk.get('source_type'):
            source_info += f" ({chunk['source_type']})"

        context_parts.append(f"{source_info}:\n{chunk['chunk_text']}")

        if chunk.get('source_url'):
            citations.append({
                'index': i + 1,
                'url': chunk['source_url'],
                'source_name': chunk.get('source_name', 'Unknown'),
                'source_type': chunk.get('source_type', 'document'),
                'relevance': chunk.get('similarity', 0)
            })

    context = "\n\n---\n\n".join(context_parts)

    # Build prompt
    system_prompt = """You are an AI product research assistant. Your task is to answer questions
based on the provided context from app reviews, changelogs, and other product research sources.

Guidelines:
- Answer based ONLY on the provided context
- Cite your sources using [Source N] notation
- If the context doesn't contain relevant information, say so
- Focus on actionable insights for product managers
- Be specific and provide examples from the sources when possible
- Highlight pain points, feature requests, and user sentiment"""

    if research_goal:
        system_prompt += f"\n\nResearch Goal: {research_goal}"

    user_prompt = f"""Context from research corpus:

{context}

---

Question: {question}

Please provide a detailed answer with citations."""

    # Call OpenAI API
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }

    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        answer = response.json()["choices"][0]["message"]["content"]

        return {
            'answer': answer,
            'citations': citations,
            'confidence': 'high' if len(context_chunks) >= 3 else 'medium' if context_chunks else 'low',
            'num_sources': len(context_chunks)
        }

    except Exception as e:
        return {
            'answer': f"Error generating answer: {str(e)}",
            'citations': [],
            'confidence': 'error',
            'num_sources': 0
        }


def answer_question(question: str, vectors: List[Dict],
                    research_goal: str = "", top_k: int = 5) -> Dict:
    """
    Full RAG pipeline: embed question, find similar chunks, generate answer.

    Args:
        question: The user's question
        vectors: All vectors from the session
        research_goal: The original research goal
        top_k: Number of chunks to retrieve

    Returns:
        Dict with answer and metadata
    """
    # Get question embedding
    question_embedding = get_embedding(question)

    # Find similar chunks
    similar_chunks = find_similar_chunks(question_embedding, vectors, top_k=top_k)

    if not similar_chunks:
        return {
            'answer': "I couldn't find any relevant information in the research corpus to answer your question. "
                      "Try rephrasing your question or ensure the session has ingested relevant data.",
            'citations': [],
            'confidence': 'none',
            'num_sources': 0
        }

    # Generate answer
    return generate_answer(question, similar_chunks, research_goal)


def summarize_insights(insights: List[Dict], research_goal: str = "") -> str:
    """
    Generate a summary of pain points and insights.

    Args:
        insights: List of insight dicts from the database
        research_goal: The original research goal

    Returns:
        Summary text
    """
    if not insights:
        return "No insights have been generated yet. Run the analysis first."

    # Build summary prompt
    insights_text = "\n".join([
        f"- {i['label']}: frequency={i['frequency']}, sentiment={i['sentiment_score']:.2f}"
        for i in insights[:20]
    ])

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }

    prompt = f"""Based on the following pain points extracted from user reviews and feedback,
provide a brief executive summary for a product manager.

Research Goal: {research_goal}

Top Pain Points (by frequency):
{insights_text}

Provide:
1. A 2-3 sentence executive summary
2. Top 3 actionable recommendations
3. Key themes identified"""

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a product research analyst providing actionable insights."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 500
    }

    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating summary: {str(e)}"
