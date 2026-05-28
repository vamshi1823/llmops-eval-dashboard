import re
from typing import Dict, List


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Simple word overlap similarity between two texts (0 to 1).
    Higher = more similar.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

def calculate_faithfulness(generated_text: str, source_text: str) -> float:
    """
    Measure how much of the generated answer is grounded in the source.
    Returns 0-1 score.
    """
    similarity = calculate_similarity(generated_text, source_text)
    return min(1.0, similarity * 1.2)  # Slight boost for partial matches

def calculate_hallucination_rate(generated_text: str, source_text: str) -> float:
    """
    Estimate hallucination (0-1, lower is better).
    High hallucination = text that's not grounded in source.
    """
    faithfulness = calculate_faithfulness(generated_text, source_text)
    return 1.0 - faithfulness

def calculate_answer_relevancy(generated_text: str, question: str) -> float:
    """
    Measure how relevant the answer is to the question.
    Returns 0-1 score.
    """
    return calculate_similarity(generated_text, question)

def evaluate_response(
    question: str,
    generated_answer: str,
    expected_answer: str,
    latency_ms: float
) -> Dict[str, float]:
    """
    Comprehensive evaluation of an LLM response.
    Returns dict with all metrics.
    """
    
    metrics = {
        "faithfulness": calculate_faithfulness(generated_answer, expected_answer),
        "hallucination_rate": calculate_hallucination_rate(generated_answer, expected_answer),
        "answer_relevancy": calculate_answer_relevancy(generated_answer, question),
        "latency_ms": latency_ms
    }
    
    # Simple F1-like score combining all metrics
    avg_quality = (metrics["faithfulness"] + metrics["answer_relevancy"]) / 2
    metrics["overall_score"] = avg_quality * (1 - min(metrics["hallucination_rate"], 0.5))
    
    return metrics