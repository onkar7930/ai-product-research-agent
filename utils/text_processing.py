"""
Text processing utilities.
Handles text chunking, normalization, and n-gram analysis.
"""

import re
from typing import List, Dict, Tuple
from collections import Counter


def normalize_text(text: str) -> str:
    """Normalize text by cleaning whitespace and special characters."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special unicode characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of dicts with 'text' and 'index' keys
    """
    text = normalize_text(text)

    if len(text) <= chunk_size:
        return [{'text': text, 'index': 0}]

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence boundary near the end
            sentence_end = text.rfind('. ', start + chunk_size - 100, end + 50)
            if sentence_end > start:
                end = sentence_end + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                'text': chunk_text,
                'index': index
            })
            index += 1

        # Move start position, accounting for overlap
        start = end - overlap if end < len(text) else len(text)

    return chunks


def extract_ngrams(text: str, n: int = 2) -> List[str]:
    """Extract n-grams from text."""
    # Tokenize: split on whitespace and punctuation
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Remove common stop words
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
        'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'this', 'that',
        'with', 'they', 'from', 'will', 'would', 'there', 'their', 'what', 'about',
        'which', 'when', 'make', 'like', 'just', 'over', 'such', 'into', 'very',
        'some', 'could', 'them', 'than', 'other', 'only', 'come', 'its', 'also',
        'back', 'after', 'use', 'how', 'your', 'well', 'way', 'even', 'want',
        'because', 'any', 'these', 'give', 'most', 'app', 'apps', 'really', 'dont',
        'get', 'got', 'using', 'used', 'does', 'doesnt'
    }

    words = [w for w in words if w not in stop_words]

    if len(words) < n:
        return []

    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])
        ngrams.append(ngram)

    return ngrams


def analyze_sentiment(text: str) -> Tuple[float, str]:
    """
    Simple rule-based sentiment analysis.
    Returns (score, label) where score is -1 to 1 and label is 'negative'/'neutral'/'positive'.
    """
    text_lower = text.lower()

    # Negative indicators (pain points)
    negative_words = [
        'problem', 'issue', 'bug', 'crash', 'slow', 'annoying', 'frustrating',
        'terrible', 'awful', 'horrible', 'hate', 'worst', 'broken', 'useless',
        'disappointed', 'disappointing', 'waste', 'fail', 'failed', 'failure',
        'difficult', 'hard', 'confusing', 'complicated', 'missing', 'lack',
        'doesn\'t work', 'won\'t', 'can\'t', 'cannot', 'unable', 'poor',
        'bad', 'sucks', 'ridiculous', 'impossible', 'nightmare', 'pain',
        'error', 'glitch', 'freeze', 'freezes', 'laggy', 'unresponsive'
    ]

    # Positive indicators
    positive_words = [
        'love', 'great', 'awesome', 'amazing', 'excellent', 'perfect', 'best',
        'fantastic', 'wonderful', 'helpful', 'easy', 'simple', 'intuitive',
        'beautiful', 'fast', 'smooth', 'reliable', 'recommend', 'useful',
        'good', 'nice', 'enjoy', 'happy', 'satisfied', 'works well'
    ]

    neg_count = sum(1 for word in negative_words if word in text_lower)
    pos_count = sum(1 for word in positive_words if word in text_lower)

    total = neg_count + pos_count
    if total == 0:
        return 0.0, 'neutral'

    score = (pos_count - neg_count) / total

    if score < -0.2:
        label = 'negative'
    elif score > 0.2:
        label = 'positive'
    else:
        label = 'neutral'

    return score, label


def extract_pain_points(texts: List[str], source_urls: List[str] = None,
                        min_frequency: int = 2) -> List[Dict]:
    """
    Extract pain point clusters from a list of texts using n-gram frequency.

    Args:
        texts: List of text documents
        source_urls: Corresponding source URLs for evidence
        min_frequency: Minimum frequency for a pain point to be included

    Returns:
        List of pain point dicts with label, frequency, sentiment, evidence
    """
    if source_urls is None:
        source_urls = [None] * len(texts)

    # Collect all bigrams and trigrams with their sources
    ngram_sources = {}  # ngram -> [(text_idx, source_url)]
    ngram_samples = {}  # ngram -> [sample_texts]

    for idx, text in enumerate(texts):
        bigrams = extract_ngrams(text, 2)
        trigrams = extract_ngrams(text, 3)

        for ngram in bigrams + trigrams:
            if ngram not in ngram_sources:
                ngram_sources[ngram] = []
                ngram_samples[ngram] = []

            ngram_sources[ngram].append((idx, source_urls[idx]))

            # Store sample text (truncated)
            sample = text[:200] + '...' if len(text) > 200 else text
            if sample not in ngram_samples[ngram]:
                ngram_samples[ngram].append(sample)

    # Filter by minimum frequency
    pain_points = []

    for ngram, sources in ngram_sources.items():
        if len(sources) >= min_frequency:
            # Calculate sentiment for this pain point
            sample_texts = ngram_samples[ngram][:5]  # Limit to 5 samples
            combined_text = ' '.join(sample_texts)
            sentiment_score, _ = analyze_sentiment(combined_text)

            # Get unique evidence URLs
            evidence_urls = list(set(url for _, url in sources if url))

            pain_points.append({
                'label': ngram,
                'frequency': len(sources),
                'sentiment_score': sentiment_score,
                'evidence_urls': evidence_urls[:10],  # Limit to 10 URLs
                'sample_texts': sample_texts
            })

    # Sort by frequency descending, then by negative sentiment
    pain_points.sort(key=lambda x: (x['frequency'], -x['sentiment_score']), reverse=True)

    return pain_points[:50]  # Return top 50 pain points


def identify_pain_point_keywords() -> List[str]:
    """Return keywords that indicate pain points in reviews."""
    return [
        'bug', 'crash', 'slow', 'problem', 'issue', 'error', 'fail', 'broken',
        'annoying', 'frustrating', 'confusing', 'difficult', 'missing', 'need',
        'wish', 'should', 'could', 'please', 'fix', 'improve', 'add', 'want',
        'hate', 'worst', 'terrible', 'awful', 'useless', 'disappointed'
    ]
