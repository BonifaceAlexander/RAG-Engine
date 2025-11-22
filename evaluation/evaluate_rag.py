"""Simple evaluation harness for RAG outputs.

This script demonstrates how one might evaluate  RAG system using a small gold dataset.
It computes simple metrics like exact-match, top-k retrieval recall, and a placeholder for RAGAS-like scores.
"""
import json
from typing import List, Dict

GOLD = [
    {"q": "What is RAG?", "a": "Retrieval-Augmented Generation"},
    {"q": "What chunk size did we use?", "a": "700"},
]

def evaluate(predict_fn):
    results = []
    for item in GOLD:
        q = item['q']
        expected = item['a'].lower()
        pred = predict_fn(q).lower()
        exact = int(expected in pred)
        results.append({'q': q, 'expected': expected, 'pred': pred, 'exact': exact})
    # compute simple accuracy
    accuracy = sum(r['exact'] for r in results)/len(results)
    print('Accuracy (contains expected):', accuracy)
    print(json.dumps(results, indent=2))
    return results

if __name__ == '__main__':
    # Example usage: provide a function that calls your engine
    def dummy_predict(q):
        # replace with engine.query(q)
        return 'Retrieval-Augmented Generation' if 'rag' in q.lower() else '700'
    evaluate(dummy_predict)
