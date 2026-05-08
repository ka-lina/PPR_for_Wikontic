from typing import List, Dict, Set
import numpy as np


class RetrievalMetrics:
    @staticmethod
    def recall_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
        """Calculate Recall@k"""
        retrieved_set = set(retrieved[:k])
        gold_set = set(gold)
        
        if len(gold_set) == 0:
            return 0.0
        
        return len(retrieved_set & gold_set) / len(gold_set)
    
    @staticmethod
    def precision_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
        """Calculate Precision@k"""
        retrieved_set = set(retrieved[:k])
        gold_set = set(gold)
        
        if k == 0:
            return 0.0
        
        return len(retrieved_set & gold_set) / k
    
    @staticmethod
    def f1_score(retrieved: List[str], gold: List[str], k: int = None) -> float:
        """Calculate F1 score (macro F1 at k if specified)"""
        if k:
            retrieved = retrieved[:k]
        
        retrieved_set = set(retrieved)
        gold_set = set(gold)
        
        if len(retrieved_set) == 0 and len(gold_set) == 0:
            return 1.0
        
        intersection = len(retrieved_set & gold_set)
        
        if intersection == 0:
            return 0.0
        
        precision = intersection / len(retrieved_set)
        recall = intersection / len(gold_set)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def evaluate_all(retrieved_per_query: List[List[str]], 
                     gold_per_query: List[List[str]], 
                     k_values: List[int] = [1, 5, 10, 20]) -> Dict:
        """Evaluate recall, precision, f1 for all k values"""
        
        results = {}
        
        for k in k_values:
            recalls = []
            precisions = []
            f1s = []
            
            for retrieved, gold in zip(retrieved_per_query, gold_per_query):
                recalls.append(RetrievalMetrics.recall_at_k(retrieved, gold, k))
                precisions.append(RetrievalMetrics.precision_at_k(retrieved, gold, k))
                f1s.append(RetrievalMetrics.f1_score(retrieved, gold, k))
            
            results[f'recall@{k}'] = np.mean(recalls)
            results[f'precision@{k}'] = np.mean(precisions)
            results[f'f1@{k}'] = np.mean(f1s)
        
        return results