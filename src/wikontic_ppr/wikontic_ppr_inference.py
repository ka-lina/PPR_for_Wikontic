# src/wikontic_ppr/wikontic_ppr_inference.py

from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import sys
import pickle

# Add Wikontic to path
sys.path.insert(0, "../Wikontic")
from wikontic.utils.inference_with_db import InferenceWithDB
from wikontic.utils.dynamic_aligner import Aligner
from wikontic.utils.openai_utils import LLMTripletExtractor

# Import HippoRAG utilities
from hipporag.utils.misc_utils import compute_mdhash_id
from hipporag.utils.embed_utils import retrieve_knn

class PassageStore:
    def __init__(self, passages_db, embedding_model, cache_dir: str = "./cache"):
        self.db = passages_db
        self.embedding_model = embedding_model
        self.collection = self.db.passages
        self.edge_collection = self.db.passage_entity_edges
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_passage_ids_path(self) -> Path:
        return self.cache_dir / "passage_ids.pickle"
    
    def save_passage_ids(self, passage_ids: List[str]):
        """Save list of all passage IDs"""
        with open(self.get_passage_ids_path(), 'wb') as f:
            pickle.dump(passage_ids, f)
    
    def load_passage_ids(self) -> Optional[List[str]]:
        """Load list of all passage IDs"""
        path = self.get_passage_ids_path()
        if not path.exists():
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def has_passage(self, passage_id: str) -> bool:
        """Check if passage already exists"""
        return self.collection.find_one({"passage_id": passage_id}) is not None
    
    def add_passage(self, text: str, source_text_id: str) -> str:
        """Add passage if not exists"""
        passage_id = compute_mdhash_id(text, prefix="passage-")
        
        if self.has_passage(passage_id):
            return passage_id
        
        embedding = self.embedding_model.encode([text])[0]
        
        self.collection.insert_one({
            "passage_id": passage_id,
            "content": text,
            "embedding": embedding.tolist(),
            "source_text_id": source_text_id
        })
        
        return passage_id
    
    def add_passage_entity_edge(self, passage_id: str, entity_name: str, entity_type: Optional[str] = None):
        """Link a passage to an entity mentioned in it"""
        self.edge_collection.update_one(
            {"passage_id": passage_id, "entity_name": entity_name},
            {"$set": {
                "entity_type": entity_type,
                "weight": 1.0
            }},
            upsert=True
        )
    
    def get_passages_for_entities(self, entity_names: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """Retrieve passages by entity names (for PPR initialization)"""
        pipeline = [
            {"$match": {"entity_name": {"$in": entity_names}}},
            {"$group": {
                "_id": "$passage_id",
                "score": {"$sum": "$weight"}
            }},
            {"$sort": {"score": -1}},
            {"$limit": top_k}
        ]
        results = list(self.edge_collection.aggregate(pipeline))
        return [(r["_id"], r["score"]) for r in results]


class WikonticPPRInference(InferenceWithDB):
    """
    Extends Wikontic's InferenceWithDB to add PPR features:
    - Passage nodes
    - Passage-entity edges
    - Embeddings for passages
    """
    
    def __init__(self, extractor, aligner, triplets_db, embedding_model, passages_db=None, cache_dir: str = "./cache"):
        super().__init__(extractor, aligner, triplets_db)
        
        self.embedding_model = embedding_model
        self.passage_store = PassageStore(passages_db or triplets_db, embedding_model, cache_dir=cache_dir)
        
    def extract_triplets_and_add_to_db_with_passages(self, text: str, source_text_id: str):
        """
        Extended version: extracts triplets AND creates passage nodes/edges
        """
        # Step 1: Create passage node FIRST
        passage_id = self.passage_store.add_passage(text, source_text_id)
        
        # Step 2: Extract triplets using parent method (non-ontology)
        initial_triplets, final_triplets, filtered_triplets = \
            self.extract_triplets(text, sample_id=None, source_text_id=source_text_id)
        
        # Step 3: Add to database using parent methods
        if len(initial_triplets) > 0:
            self.aligner.add_initial_triplets(initial_triplets, sample_id=None)
        if len(final_triplets) > 0:
            self.aligner.add_triplets(final_triplets, sample_id=None)
        if len(filtered_triplets) > 0:
            self.aligner.add_filtered_triplets(filtered_triplets, sample_id=None)
        
        # Step 4: Create passage-entity edges
        all_entities = set()
        
        for triplet in final_triplets:
            all_entities.add(triplet.get("subject", ""))
            all_entities.add(triplet.get("object", ""))
        
        for entity in all_entities:
            if entity:
                canonical = self._get_canonical_entity(entity)
                self.passage_store.add_passage_entity_edge(passage_id, canonical, None)  # No type without ontology
        
        return initial_triplets, final_triplets, filtered_triplets, passage_id
    
    def _get_canonical_entity(self, entity_name: str) -> str:
        """Find canonical name from entity_aliases collection"""
        collection = self.triplets_db.get_collection("entity_aliases")
        result = collection.find_one({"alias": entity_name})
        if result:
            return result.get("label", entity_name)
        
        # If not found, check if it's already canonical
        result = collection.find_one({"label": entity_name})
        if result:
            return entity_name
        
        return entity_name
    
    def _get_entity_type(self, entity_name: str, triplets: List[Dict]) -> Optional[str]:
        """Extract entity type from triplets"""
        for triplet in triplets:
            if triplet.get("subject") == entity_name:
                return triplet.get("subject_type")
            if triplet.get("object") == entity_name:
                return triplet.get("object_type")
        return None
    
    