# src/wikontic_ppr/wikontic_to_hipporag.py

from typing import List, Tuple
from pymongo import MongoClient


class WikonticToHippoRAGConverter:
    """Convert Wikontic MongoDB data to HippoRAG format"""
    
    def __init__(self, triplets_db):
        self.db = triplets_db
    
    def get_passages_from_collection(self, collection_name: str = "passages") -> Tuple[List[str], List[str]]:
        """Extract passages and their IDs from passage store"""
        collection = self.db.get_collection(collection_name)
        passages = list(collection.find())
        
        passage_texts = [p["content"] for p in passages]
        passage_ids = [p["passage_id"] for p in passages]
        
        print(f"Found {len(passage_texts)} passages")
        return passage_texts, passage_ids
    
    def get_triplets_for_passages(self, passage_ids: List[str]) -> List[List[Tuple[str, str, str]]]:
        """Get triplets for each passage from passage-entity edges and triplets"""
        
        triplets_per_passage = []
        triplets_collection = self.db.get_collection("triplets")
        edges_collection = self.db.get_collection("passage_entity_edges")
        
        for pid in passage_ids:
            # Get entities in this passage
            edges = list(edges_collection.find({"passage_id": pid}))
            entities_in_passage = [e["entity_name"] for e in edges]
            
            # Find triplets involving these entities
            passage_triplets = []
            for entity in entities_in_passage:
                triplets = list(triplets_collection.find({
                    "$or": [
                        {"subject": entity},
                        {"object": entity}
                    ]
                }))
                
                for t in triplets:
                    triplet = (t["subject"], t["relation"], t["object"])
                    if triplet not in passage_triplets:
                        passage_triplets.append(triplet)
            
            triplets_per_passage.append(passage_triplets)
            print(f"Passage {pid[:16]}...: {len(passage_triplets)} triplets")
        
        return triplets_per_passage
    
    def get_all_triplets(self) -> List[Tuple[str, str, str]]:
        """Get all unique triplets from collection"""
        collection = self.db.get_collection("triplets")
        triplets = list(collection.find())
        
        unique_triplets = []
        for t in triplets:
            triplet = (t["subject"], t["relation"], t["object"])
            if triplet not in unique_triplets:
                unique_triplets.append(triplet)
        
        return unique_triplets