# src/data/hotpotqa_loader.py

import json
from typing import List, Dict, Tuple
from pathlib import Path


class HotPotQALoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load(self) -> Tuple[List[str], List[str], List[str], List[List[str]], List[List[str]]]:
        """
        Load HotPotQA dataset with all data aligned by question index.
        
        Returns:
            documents: List of all unique context passages
            queries: List of questions (index aligned with gold_passages)
            answers: List of answers (index aligned with queries)
            gold_passages: List of lists of passage IDs that contain the answer
            question_contexts: List of lists of context passages for each question
        """
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        queries = []
        answers = []
        gold_passages = []
        question_contexts = []
        all_documents = {}
        
        for item in data:
            # Extract question and answer
            question = item["question"]
            answer = item["answer"]
            queries.append(question)
            answers.append(answer)
            
            # Extract context passages for this question
            contexts = []
            gold_ids = []
            
            for ctx_title, ctx_sentences in item["context"]:
                # Join sentences into one passage
                passage_text = " ".join(ctx_sentences)
                passage_id = f"{ctx_title}"
                
                contexts.append(passage_text)
                all_documents[passage_id] = passage_text
                
                # Check if this context is a supporting fact
                for supporting_fact in item.get("supporting_facts", []):
                    if supporting_fact[0] == ctx_title:
                        gold_ids.append(passage_id)
            
            question_contexts.append(contexts)
            gold_passages.append(gold_ids)
        
        # Convert all_documents to list
        documents = list(all_documents.values())
        passage_id_to_idx = {pid: i for i, pid in enumerate(all_documents.keys())}
        
        # Convert gold passage titles to indices
        gold_passage_indices = []
        for gold_ids in gold_passages:
            indices = [passage_id_to_idx[pid] for pid in gold_ids if pid in passage_id_to_idx]
            gold_passage_indices.append(indices)
        
        return documents, queries, answers, gold_passage_indices, question_contexts