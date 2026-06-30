from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        import sys
        import asyncio
        if sys.platform == "win32":
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except Exception:
                pass

        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        import numpy as np
        
        from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        if not LLM_API_KEY:
            raise ValueError("No API key configured for LLM.")

        if LLM_BASE_URL:
            llm = ChatOpenAI(model=LLM_MODEL, openai_api_key=LLM_API_KEY, openai_api_base=LLM_BASE_URL, temperature=0.0)
        else:
            llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=LLM_API_KEY, temperature=0.0)

        # Use local embeddings to avoid cost and connection issues for Ragas
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        df = result.to_pandas()

        def safe_float(val):
            try:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return 0.0
                return float(val)
            except Exception:
                return 0.0

        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=row["contexts"],
                ground_truth=row["ground_truth"],
                faithfulness=safe_float(row.get("faithfulness", 0.0)),
                answer_relevancy=safe_float(row.get("answer_relevancy", 0.0)),
                context_precision=safe_float(row.get("context_precision", 0.0)),
                context_recall=safe_float(row.get("context_recall", 0.0))
            )
            for _, row in df.iterrows()
        ]
        return {
            "faithfulness": safe_float(result.get("faithfulness", 0.0)),
            "answer_relevancy": safe_float(result.get("answer_relevancy", 0.0)),
            "context_precision": safe_float(result.get("context_precision", 0.0)),
            "context_recall": safe_float(result.get("context_recall", 0.0)),
            "per_question": per_question
        }
    except Exception as e:
        print(f"  [Warning] RAGAS evaluation failed: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [
                EvalResult(
                    question=q, answer=a, contexts=c, ground_truth=gt,
                    faithfulness=0.0, answer_relevancy=0.0, context_precision=0.0, context_recall=0.0
                )
                for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
            ]
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    metrics_list = []
    for r in eval_results:
        scores = {
            "faithfulness": r.faithfulness,
            "context_recall": r.context_recall,
            "context_precision": r.context_precision,
            "answer_relevancy": r.answer_relevancy
        }
        avg_score = sum(scores.values()) / 4.0
        worst_metric = min(scores, key=scores.get)
        worst_score = scores[worst_metric]
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        metrics_list.append({
            "question": r.question,
            "worst_metric": worst_metric,
            "score": worst_score,
            "avg_score": avg_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix
        })
    
    metrics_list = sorted(metrics_list, key=lambda x: x["avg_score"])
    return [
        {
            "question": item["question"],
            "worst_metric": item["worst_metric"],
            "score": item["score"],
            "diagnosis": item["diagnosis"],
            "suggested_fix": item["suggested_fix"]
        }
        for item in metrics_list[:bottom_n]
    ]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
