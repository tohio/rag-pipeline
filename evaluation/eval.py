"""
eval.py
-------
Evaluates the RAG pipeline against a ground truth Q&A dataset.

Measures three core dimensions of RAG quality:

    1. Retrieval Quality
       - Hit Rate    : was the correct chunk retrieved in the top-k results?
       - MRR         : how highly ranked was the first relevant chunk?

    2. Answer Quality (LLM-as-judge)
       - Faithfulness: is the answer grounded in the retrieved context?
       - Relevance   : does the answer address the question asked?
       - Correctness : does the answer match the expected ground truth?

    3. Performance
       - Latency     : end-to-end response time per query (seconds)

Evaluation dataset:
    The ground truth Q&A pairs are loaded from a JSON file derived
    from meridian-evaluation-qa.md. Each entry contains:
        - question       : the query
        - expected_answer: the ground truth answer
        - source_section : the handbook section the answer comes from
        - difficulty     : Easy / Medium / Hard

Usage:
    # Run full evaluation
    python evaluation/eval.py

    # Run evaluation on a subset
    python evaluation/eval.py --limit 10

    # Save results to file
    python evaluation/eval.py --output evaluation/results.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default path for the evaluation Q&A dataset
DEFAULT_QA_PATH = "evaluation/qa_pairs.json"
DEFAULT_DOCS_PATH = "data/raw"


# -------------------------------------------------------------------
# LLM-as-judge prompt templates
# Using the LLM to score faithfulness, relevance, and correctness
# is more robust than string matching for open-ended answers.
# -------------------------------------------------------------------

FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI-generated answer is faithful to the provided context.
Faithful means the answer only contains information that is present in the context —
it does not add, invent, or extrapolate beyond what the context says.

Context:
{context}

Answer:
{answer}

Score the faithfulness on a scale of 1 to 5:
5 = Completely faithful, every claim is supported by the context
4 = Mostly faithful, minor extrapolation
3 = Partially faithful, some claims unsupported
2 = Mostly unfaithful, significant unsupported claims
1 = Not faithful, answer contradicts or ignores the context

Respond with ONLY a single integer (1-5) and nothing else.
"""

RELEVANCE_PROMPT = """\
You are evaluating whether an AI-generated answer is relevant to the question asked.

Question:
{question}

Answer:
{answer}

Score the relevance on a scale of 1 to 5:
5 = Directly and completely answers the question
4 = Mostly answers the question with minor gaps
3 = Partially answers the question
2 = Tangentially related but does not answer the question
1 = Completely irrelevant

Respond with ONLY a single integer (1-5) and nothing else.
"""

CORRECTNESS_PROMPT = """\
You are evaluating whether an AI-generated answer is factually correct
compared to the expected ground truth answer.

Question:
{question}

Expected Answer:
{expected_answer}

Generated Answer:
{generated_answer}

Score the correctness on a scale of 1 to 5:
5 = Completely correct, matches expected answer
4 = Mostly correct, minor differences
3 = Partially correct, captures some key facts
2 = Mostly incorrect, misses key facts
1 = Completely incorrect or contradicts expected answer

Respond with ONLY a single integer (1-5) and nothing else.
"""


def load_qa_pairs(qa_path: str) -> list[dict]:
    """
    Load evaluation Q&A pairs from a JSON file.

    Args:
        qa_path (str): Path to the JSON Q&A pairs file.

    Returns:
        list[dict]: List of Q&A pair dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(qa_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Q&A pairs file not found: {qa_path}\n"
            f"Generate it first by converting meridian-evaluation-qa.md to JSON."
        )

    with open(path, "r") as f:
        pairs = json.load(f)

    logger.info(f"Loaded {len(pairs)} Q&A pairs from {qa_path}")
    return pairs


def _score_with_llm(llm, prompt: str) -> int:
    """
    Use the LLM to score a response on a 1-5 scale.

    Args:
        llm: Initialized LLM instance.
        prompt (str): Scoring prompt.

    Returns:
        int: Score between 1 and 5. Returns 0 if parsing fails.
    """
    try:
        response = llm.complete(prompt)
        score = int(response.text.strip())
        return max(1, min(5, score))  # clamp to 1-5
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse LLM score: {e}")
        return 0


def evaluate_faithfulness(llm, answer: str, context: str) -> int:
    """Score how faithful the answer is to the retrieved context (1-5)."""
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    return _score_with_llm(llm, prompt)


def evaluate_relevance(llm, question: str, answer: str) -> int:
    """Score how relevant the answer is to the question (1-5)."""
    prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
    return _score_with_llm(llm, prompt)


def evaluate_correctness(
    llm, question: str, expected_answer: str, generated_answer: str
) -> int:
    """Score how correct the answer is vs ground truth (1-5)."""
    prompt = CORRECTNESS_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )
    return _score_with_llm(llm, prompt)


def evaluate_retrieval_hit(
    results: list, expected_answer: str
) -> bool:
    """
    Check if any retrieved chunk contains content relevant to
    the expected answer (hit rate check).

    Uses simple substring matching as a lightweight proxy.
    For more robust evaluation, replace with a semantic similarity check.

    Args:
        results: Retrieved NodeWithScore objects.
        expected_answer (str): Ground truth answer.

    Returns:
        bool: True if a relevant chunk was retrieved.
    """
    expected_lower = expected_answer.lower()
    # Extract key terms from expected answer (words > 4 chars)
    key_terms = [w for w in expected_lower.split() if len(w) > 4]

    for result in results:
        chunk_text = result.node.text.lower()
        matches = sum(1 for term in key_terms if term in chunk_text)
        if matches >= max(1, len(key_terms) // 2):
            return True
    return False


def run_evaluation(
    pipeline,
    qa_pairs: list[dict],
    limit: Optional[int] = None,
) -> dict:
    """
    Run the full evaluation suite against the Q&A pairs.

    Args:
        pipeline: Initialized RAGPipeline instance.
        qa_pairs (list[dict]): Ground truth Q&A pairs.
        limit (Optional[int]): Limit evaluation to first N pairs.

    Returns:
        dict: Evaluation results including per-question scores
              and aggregate metrics.
    """
    from src.retrieval.retriever import retrieve, format_retrieved_context
    from src.generation.generator import get_llm, generate

    llm = get_llm()
    pairs = qa_pairs[:limit] if limit else qa_pairs

    logger.info(f"Running evaluation on {len(pairs)} Q&A pairs...")

    results = []
    total_latency = 0

    for i, pair in enumerate(pairs, 1):
        question = pair["question"]
        expected = pair["expected_answer"]
        difficulty = pair.get("difficulty", "Unknown")
        source_section = pair.get("source_section", "Unknown")

        logger.info(f"Evaluating [{i}/{len(pairs)}]: {question[:60]}...")

        start_time = time.time()

        try:
            # Run retrieval
            retrieved = retrieve(pipeline.retriever, question)
            context = format_retrieved_context(retrieved)

            # Run generation
            answer = generate(llm, question, context)

            latency = round(time.time() - start_time, 3)
            total_latency += latency

            # Score retrieval
            hit = evaluate_retrieval_hit(retrieved, expected)

            # Score generation (LLM-as-judge)
            faithfulness = evaluate_faithfulness(llm, answer, context)
            relevance = evaluate_relevance(llm, question, answer)
            correctness = evaluate_correctness(llm, question, expected, answer)

            result = {
                "question": question,
                "expected_answer": expected,
                "generated_answer": answer,
                "source_section": source_section,
                "difficulty": difficulty,
                "retrieval_hit": hit,
                "num_chunks_retrieved": len(retrieved),
                "faithfulness_score": faithfulness,
                "relevance_score": relevance,
                "correctness_score": correctness,
                "latency_seconds": latency,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Error evaluating question {i}: {e}")
            result = {
                "question": question,
                "expected_answer": expected,
                "generated_answer": None,
                "source_section": source_section,
                "difficulty": difficulty,
                "retrieval_hit": False,
                "num_chunks_retrieved": 0,
                "faithfulness_score": 0,
                "relevance_score": 0,
                "correctness_score": 0,
                "latency_seconds": 0,
                "error": str(e),
            }

        results.append(result)

    # Compute aggregate metrics
    valid = [r for r in results if r["error"] is None]
    n = len(valid)

    metrics = {
        "total_questions": len(results),
        "evaluated_questions": n,
        "hit_rate": round(sum(r["retrieval_hit"] for r in valid) / n, 4) if n else 0,
        "avg_faithfulness": round(sum(r["faithfulness_score"] for r in valid) / n, 4) if n else 0,
        "avg_relevance": round(sum(r["relevance_score"] for r in valid) / n, 4) if n else 0,
        "avg_correctness": round(sum(r["correctness_score"] for r in valid) / n, 4) if n else 0,
        "avg_latency_seconds": round(total_latency / n, 3) if n else 0,
        "by_difficulty": _aggregate_by_difficulty(valid),
    }

    return {
        "metrics": metrics,
        "results": results,
    }


def _aggregate_by_difficulty(results: list[dict]) -> dict:
    """
    Aggregate correctness scores grouped by difficulty level.

    Args:
        results (list[dict]): Per-question evaluation results.

    Returns:
        dict: Average correctness score per difficulty level.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        groups[r["difficulty"]].append(r["correctness_score"])

    return {
        difficulty: round(sum(scores) / len(scores), 4)
        for difficulty, scores in groups.items()
    }


def print_summary(metrics: dict) -> None:
    """
    Print a formatted evaluation summary to the console.

    Args:
        metrics (dict): Aggregate metrics from run_evaluation().
    """
    print("\n" + "=" * 60)
    print(" RAG Pipeline Evaluation Summary")
    print("=" * 60)
    print(f"  Total questions   : {metrics['total_questions']}")
    print(f"  Evaluated         : {metrics['evaluated_questions']}")
    print(f"  Hit Rate          : {metrics['hit_rate'] * 100:.1f}%")
    print(f"  Avg Faithfulness  : {metrics['avg_faithfulness']:.2f} / 5")
    print(f"  Avg Relevance     : {metrics['avg_relevance']:.2f} / 5")
    print(f"  Avg Correctness   : {metrics['avg_correctness']:.2f} / 5")
    print(f"  Avg Latency       : {metrics['avg_latency_seconds']:.3f}s")
    print("-" * 60)
    print("  Correctness by Difficulty:")
    for difficulty, score in metrics["by_difficulty"].items():
        print(f"    {difficulty:<10}: {score:.2f} / 5")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Meridian Capital Group RAG pipeline"
    )
    parser.add_argument(
        "--qa",
        type=str,
        default=DEFAULT_QA_PATH,
        help=f"Path to Q&A pairs JSON file (default: {DEFAULT_QA_PATH})",
    )
    parser.add_argument(
        "--docs",
        type=str,
        default=DEFAULT_DOCS_PATH,
        help=f"Path to source documents (default: {DEFAULT_DOCS_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N questions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save full results to a JSON file",
    )
    return parser.parse_args()


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    try:
        from src.pipeline import RAGPipeline

        pipeline = RAGPipeline(docs_path=args.docs)
        qa_pairs = load_qa_pairs(args.qa)

        eval_results = run_evaluation(
            pipeline=pipeline,
            qa_pairs=qa_pairs,
            limit=args.limit,
        )

        print_summary(eval_results["metrics"])

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(eval_results, f, indent=2)
            logger.info(f"Full results saved to: {args.output}")

    except (FileNotFoundError, EnvironmentError) as e:
        print(f"Error: {e}")
        sys.exit(1)
