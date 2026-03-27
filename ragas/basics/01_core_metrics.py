# RAGAS Basics: Core Metrics & Dataset Creation
# Use Case: Evaluate a Customer Support RAG System

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

load_dotenv()

# --- 1. RAGAS Core Metrics Explained ---
"""
RAGAS evaluates RAG pipelines on 4 key dimensions:

1. FAITHFULNESS       - Is the answer grounded in the retrieved context? (0-1)
                        Detects hallucinations. High = no hallucination.

2. ANSWER RELEVANCY   - Is the answer relevant to the question? (0-1)
                        Penalizes incomplete or off-topic answers.

3. CONTEXT PRECISION  - Are retrieved chunks relevant? (0-1)
                        High = retriever returns only useful chunks.

4. CONTEXT RECALL     - Does context contain all needed info? (0-1)
                        High = retriever doesn't miss important chunks.
"""

# --- 2. Evaluation Dataset ---
# Format: question, answer (from RAG), contexts (retrieved), ground_truth
eval_data = {
    "question": [
        "What is the refund policy?",
        "How do I contact customer support?",
        "What does the premium plan include?",
        "How do I reset my password?",
    ],
    "answer": [
        "You can return items within 30 days of purchase with the original receipt. Digital products are non-refundable.",
        "Customer support is available 24/7 via chat. Phone support is available Monday to Friday, 9AM to 6PM EST.",
        "The premium plan costs $9.99/month and includes unlimited storage, priority support, and advanced analytics.",
        "Go to Settings, then Security, then click Reset Password. You will receive an email within 5 minutes.",
    ],
    "contexts": [
        ["Our refund policy allows returns within 30 days of purchase with original receipt. Digital products are non-refundable."],
        ["Our customer support is available 24/7 via chat. Phone support is available Mon-Fri 9AM-6PM EST."],
        ["Premium subscription costs $9.99/month or $99/year. It includes unlimited storage, priority support, and advanced analytics."],
        ["To reset your password: go to Settings > Security > Reset Password. You'll receive an email within 5 minutes."],
    ],
    "ground_truth": [
        "Items can be returned within 30 days with receipt. Digital products cannot be refunded.",
        "24/7 chat support is available. Phone support runs Monday-Friday 9AM-6PM EST.",
        "Premium is $9.99/month and includes unlimited storage, priority support, and advanced analytics.",
        "Reset password via Settings > Security > Reset Password. Email arrives within 5 minutes.",
    ]
}

dataset = Dataset.from_dict(eval_data)
print(f"Evaluation dataset: {len(dataset)} samples")

# --- 3. Run RAGAS Evaluation ---
print("\n=== Running RAGAS Evaluation ===")
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print("\n=== RAGAS Scores ===")
print(results)

# --- 4. Per-Sample Analysis ---
df = results.to_pandas()
print("\n=== Per-Question Scores ===")
print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]].to_string())

# --- 5. Identify Weak Points ---
print("\n=== Weakest Performing Questions ===")
for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
    if metric in df.columns:
        worst = df.loc[df[metric].idxmin()]
        print(f"\n{metric}: {worst[metric]:.3f}")
        print(f"  Question: {worst['question']}")
