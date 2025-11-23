"""
OpenAI Embeddings utilities.
Handles text embedding with caching to minimize API costs.
"""

import os
import requests
from typing import List, Dict, Any
from .database import get_cached_embedding, save_cached_embedding


OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_URL = "https://api.openai.com/v1/embeddings"


def get_embedding(text: str, use_cache: bool = True) -> List[float]:
    """
    Get embedding for a single text using OpenAI API.
    Uses caching to minimize API costs.
    """
    # Check cache first
    if use_cache:
        cached = get_cached_embedding(text)
        if cached:
            return cached

    # Call OpenAI API
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENAI_EMBEDDING_MODEL,
        "input": text,
        "encoding_format": "float"
    }

    response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
    response.raise_for_status()

    embedding = response.json()["data"][0]["embedding"]

    # Cache the result
    if use_cache:
        save_cached_embedding(text, embedding)

    return embedding


def get_embeddings_batch(texts: List[str], use_cache: bool = True, batch_size: int = 100) -> List[List[float]]:
    """
    Get embeddings for multiple texts.
    Checks cache first, then batches uncached texts for API efficiency.
    """
    results = [None] * len(texts)
    uncached_indices = []
    uncached_texts = []

    # Check cache for each text
    if use_cache:
        for i, text in enumerate(texts):
            cached = get_cached_embedding(text)
            if cached:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
    else:
        uncached_indices = list(range(len(texts)))
        uncached_texts = texts

    # Batch process uncached texts
    if uncached_texts:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }

        for batch_start in range(0, len(uncached_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(uncached_texts))
            batch_texts = uncached_texts[batch_start:batch_end]

            payload = {
                "model": OPENAI_EMBEDDING_MODEL,
                "input": batch_texts,
                "encoding_format": "float"
            }

            response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
            response.raise_for_status()

            embeddings_data = response.json()["data"]

            # Sort by index to maintain order
            embeddings_data.sort(key=lambda x: x["index"])

            for j, emb_data in enumerate(embeddings_data):
                original_idx = uncached_indices[batch_start + j]
                embedding = emb_data["embedding"]
                results[original_idx] = embedding

                # Cache the result
                if use_cache:
                    save_cached_embedding(texts[original_idx], embedding)

    return results


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def find_similar_chunks(query_embedding: List[float], vectors: List[Dict],
                        top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
    """
    Find the most similar chunks to a query embedding.
    Returns chunks sorted by similarity score.
    """
    import json

    scored_chunks = []

    for vec in vectors:
        # Parse embedding from JSON if needed
        embedding = vec['embedding']
        if isinstance(embedding, str):
            embedding = json.loads(embedding)

        similarity = cosine_similarity(query_embedding, embedding)

        if similarity >= threshold:
            scored_chunks.append({
                'chunk_text': vec['chunk_text'],
                'source_url': vec.get('source_url'),
                'source_name': vec.get('source_name'),
                'source_type': vec.get('source_type'),
                'similarity': similarity,
                'metadata': vec.get('metadata', {})
            })

    # Sort by similarity descending
    scored_chunks.sort(key=lambda x: x['similarity'], reverse=True)

    return scored_chunks[:top_k]
