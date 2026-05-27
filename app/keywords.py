import re
from collections import Counter
from typing import List, Set
import string

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "that", "this", "these", "those", "it", "its", "we", "our", "you",
    "your", "they", "their", "i", "me", "my", "he", "she", "his", "her",
    "experience", "work", "working", "team", "role", "strong", "good",
    "ability", "skills", "skill", "knowledge", "including", "etc", "also",
    "well", "such", "must", "not", "more", "other", "new", "all", "into",
    "about", "up", "out", "if", "when", "what", "which", "who", "how",
    "than", "then", "so", "no", "any", "each", "both", "few", "most",
    "own", "same", "great", "excellent", "within", "across", "between",
    "during", "through", "while", "after", "before", "over", "under",
}

TECH_BIGRAMS = {
    "machine learning", "deep learning", "natural language", "language processing",
    "computer vision", "neural network", "neural networks", "data science",
    "software engineering", "software development", "object detection",
    "convolutional neural", "large language", "language model", "language models",
    "real time", "real-time", "version control", "continuous integration",
    "continuous deployment", "ci cd", "restful api", "rest api", "api design",
    "unit testing", "test driven", "agile development", "cloud computing",
    "distributed systems", "microservices architecture", "vector database",
    "transfer learning", "fine tuning", "prompt engineering",
}


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s\-\+\#]", " ", text)
    tokens = text.split()
    return [t.strip(string.punctuation) for t in tokens if t.strip(string.punctuation)]


def _extract_bigrams(text: str) -> List[str]:
    found = []
    lower = text.lower()
    for bigram in TECH_BIGRAMS:
        if bigram in lower:
            found.append(bigram)
    return found


def extract_keywords(text: str, top_n: int = 30) -> List[str]:
    tokens = _tokenize(text)
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 2 and not t.isdigit()]
    freq = Counter(filtered)
    bigrams = _extract_bigrams(text)
    top_single = [word for word, _ in freq.most_common(top_n)]
    combined = list(dict.fromkeys(bigrams + top_single))
    return combined[:top_n]


def keyword_overlap(resume_kws: List[str], job_kws: List[str]):
    resume_set: Set[str] = set(resume_kws)
    job_set: Set[str] = set(job_kws)
    matched = sorted(resume_set & job_set)
    missing = sorted(job_set - resume_set)
    extra = sorted(resume_set - job_set)
    return matched, missing, extra
