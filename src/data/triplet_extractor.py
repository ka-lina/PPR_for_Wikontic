# src/data/triplet_extractor.py

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Add Wikontic to path
wikontic_path = Path("../Wikontic")  # Adjust path as needed
sys.path.insert(0, str(wikontic_path))

from wikontic.utils.openai_utils import LLMTripletExtractor
from wikontic.utils.structured_aligner import Aligner


class TripletExtractor:
    def __init__(self, llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "cuda"):
        self.extractor = LLMTripletExtractor(model_name=llm_model_name, device=device)
        
    def extract_from_documents(self, documents: List[str]) -> List[Dict]:
        """Extract triplets from all documents"""
        all_triplets = []
        
        for doc_id, doc_text in enumerate(documents):
            result = self.extractor.extract_triplets_from_text(doc_text)
            print(doc_id, result)
            all_triplets.append({
                'doc_id': doc_id,
                'text': doc_text,
                'triplets': result.get('triplets', [])
            })
        
        return all_triplets