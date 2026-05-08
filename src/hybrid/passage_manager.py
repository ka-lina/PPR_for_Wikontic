# src/hybrid/passage_manager.py

from pymongo import MongoClient
from typing import List, Dict, Optional
import numpy as np

from hipporag.utils.misc_utils import compute_mdhash_id


class PassageManager:
    """Manages passage storage and passage-entity relationships"""
    
    def __init__(self, db, embedding_model):
        self.db = db
        self.embedding_model = embedding_model
        
        # Create collections if they don't exist
        self.passages = self.db.passages
        self.passage_entity_edges = self.db.passage_entity_edges
        
        # Create indexes
        self.passages.create_index("passage_id", unique=True)
        self.passages.create_index("source_doc_id")
        self.passage_entity_edges.create_index("passage_id")
        self.passage_entity_edges.create_index("entity_name")
        
    def add_passage(self, text: str, source_doc_id: str, chunk_index: int = 0) -> str:
        """Add a single passage to the database"""
        
        # Create unique ID from content
        passage_id = compute_mdhash_id(text, prefix="passage-")
        
        # Check if already exists
        if self.passages.find_one({"passage_id": passage_id}):
            return passage_id
        
        # Create embedding
        embedding = self.embedding_model.encode([text])[0]
        
        # Store passage
        self.passages.insert_one({
            "passage_id": passage_id,
            "content": text,
            "embedding": embedding.tolist(),
            "source_doc_id": source_doc_id,
            "chunk_index": chunk_index
        })
        
        return passage_id
    
    def add_passages_from_documents(self, documents: List[str], chunk_size: int = 500):
        """Split documents into passages and add them"""
        
        all_passage_ids = []
        
        for doc_id, doc_text in enumerate(documents):
            # Split into chunks
            chunks = self._chunk_text(doc_text, chunk_size)
            
            for chunk_idx, chunk in enumerate(chunks):
                passage_id = self.add_passage(
                    text=chunk,
                    source_doc_id=str(doc_id),
                    chunk_index=chunk_idx
                )
                all_passage_ids.append(passage_id)
        
        return all_passage_ids
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into overlapping chunks"""
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def add_passage_entity_edge(self, passage_id: str, entity_name: str, 
                                 entity_type: Optional[str] = None, weight: float = 1.0):
        """Link a passage to an entity mentioned in it"""
        
        self.passage_entity_edges.update_one(
            {"passage_id": passage_id, "entity_name": entity_name},
            {"$set": {
                "entity_type": entity_type,
                "weight": weight
            }},
            upsert=True
        )
    
    def get_passage(self, passage_id: str) -> Optional[Dict]:
        """Get passage by ID"""
        return self.passages.find_one({"passage_id": passage_id})
    
    def get_passages_for_entity(self, entity_name: str, limit: int = 20) -> List[Dict]:
        """Get all passages containing an entity"""
        
        edges = self.passage_entity_edges.find({"entity_name": entity_name}).limit(limit)
        passage_ids = [e["passage_id"] for e in edges]
        
        return list(self.passages.find({"passage_id": {"$in": passage_ids}}))
    
    def get_entity_chunk_counts(self) -> Dict[str, int]:
        """Get how many passages contain each entity (for IDF weighting)"""
        
        pipeline = [
            {"$group": {"_id": "$entity_name", "count": {"$sum": 1}}}
        ]
        
        results = self.passage_entity_edges.aggregate(pipeline)
        return {r["_id"]: r["count"] for r in results}