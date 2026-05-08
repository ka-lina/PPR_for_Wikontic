# src/data/extract_and_embed.py

from typing import List, Dict, Tuple
import numpy as np
from .hotpotqa_loader import HotPotQALoader
from .triplet_extractor import TripletExtractor
from .embedding_creator import EmbeddingCreator


class DataProcessor:
    """Process HotPotQA: extract triplets and create embeddings once"""
    
    def __init__(self, llm_model: str = "Qwen/Qwen2.5-3B-Instruct", 
                 embedding_model: str = "facebook/contriever",
                 device: str = "cuda"):
        
        self.triplet_extractor = TripletExtractor(llm_model, device)
        self.embedding_creator = EmbeddingCreator(embedding_model, device)
        
    def process(self, documents: List[str]) -> Dict:
        """Process all documents once, return both Wikontic and HippoRAG formats"""
        
        # 1. Extract triplets using Wikontic's LLM extractor
        print("Extracting triplets...")
        triplet_results = self.triplet_extractor.extract_from_documents(documents)
        
        # 2. Extract canonical entities and facts from triplets
        all_entities = set()
        all_triplets = []
        
        for doc_result in triplet_results:
            for triplet in doc_result['triplets']:
                subject = triplet['subject']
                relation = triplet['relation']
                obj = triplet['object']
                
                all_entities.add(subject)
                all_entities.add(obj)
                all_triplets.append((subject, relation, obj))
        
        # 3. Create embeddings for all components
        print(f"Creating embeddings for {len(all_entities)} entities...")
        entity_embeddings = self.embedding_creator.create_entity_embeddings(list(all_entities))
        
        print(f"Creating embeddings for {len(all_triplets)} facts...")
        fact_embeddings = self.embedding_creator.create_fact_embeddings(all_triplets)
        
        print(f"Creating embeddings for {len(documents)} passages...")
        passage_embeddings = self.embedding_creator.create_passage_embeddings(documents)
        
        return {
            # For both systems
            'documents': documents,
            'triplet_results': triplet_results,
            'all_triplets': all_triplets,
            'all_entities': list(all_entities),
            
            # Embeddings
            'entity_embeddings': entity_embeddings,
            'fact_embeddings': fact_embeddings,
            'passage_embeddings': passage_embeddings,
            
            # For HippoRAG (needs hash IDs)
            'passage_ids': [compute_mdhash_id(doc, prefix="chunk-") for doc in documents],
            
            # For Wikontic (needs MongoDB structure)
            'canonical_entities': all_entities,
        }


# Helper imports
from hipporag.utils.misc_utils import compute_mdhash_id