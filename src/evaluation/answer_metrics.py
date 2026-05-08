from typing import List, Dict
import numpy as np


class AnswerMetrics:
    @staticmethod
    def exact_match(predicted: str, gold: str) -> int:
        """Exact match between predicted and gold answer"""
        return 1 if predicted.strip().lower() == gold.strip().lower() else 0
    
    @staticmethod
    def token_f1(predicted: str, gold: str) -> float:
        """Token-level F1 score"""
        pred_tokens = set(predicted.lower().split())
        gold_tokens = set(gold.lower().split())
        
        if len(pred_tokens) == 0 and len(gold_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return 0.0
        
        intersection = len(pred_tokens & gold_tokens)
        precision = intersection / len(pred_tokens)
        recall = intersection / len(gold_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def evaluate_all(predicted_answers: List[str], gold_answers: List[str]) -> Dict[str, float]:
        """Evaluate all answer metrics"""
        exact_matches = []
        f1_scores = []
        
        for pred, gold in zip(predicted_answers, gold_answers):
            exact_matches.append(AnswerMetrics.exact_match(pred, gold))
            f1_scores.append(AnswerMetrics.token_f1(pred, gold))
        
        return {
            'exact_match': np.mean(exact_matches),
            'answer_f1': np.mean(f1_scores)
        }


def calculate_answer_metrics(predicted_answers: List[str], gold_answers: List[str]) -> Dict[str, float]:
    """Convenience function"""
    return AnswerMetrics.evaluate_all(predicted_answers, gold_answers)