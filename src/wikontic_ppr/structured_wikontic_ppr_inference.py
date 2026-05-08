# src/wikontic_ppr/structured_wikontic_ppr_inference.py

import sys
sys.path.insert(0, "../Wikontic")

from wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
from wikontic.utils.structured_aligner import Aligner
from wikontic.utils.openai_utils import LLMTripletExtractor

from hipporag.utils.misc_utils import compute_mdhash_id
from .wikontic_ppr_inference import PassageStore


class StructuredWikonticPPRInference(StructuredInferenceWithDB):
    """
    Extends Wikontic's StructuredInferenceWithDB (with ontology) to add PPR features
    """
    
    def __init__(self, extractor, aligner, triplets_db, ontology_db, embedding_model, passages_db=None):
        super().__init__(extractor, aligner, triplets_db)
        
        self.ontology_db = ontology_db
        self.embedding_model = embedding_model
        self.passage_store = PassageStore(passages_db or triplets_db, embedding_model)
    
    def extract_triplets_and_add_to_db_with_passages(self, text: str, source_text_id: str, sample_id: str = None):
        """
        Extended version with ontology filtering and passage creation
        """
        # Step 1: Create passage node FIRST
        passage_id = self.passage_store.add_passage(text, source_text_id)
        
        # Step 2: Extract triplets with ontology filtering (parent method)
        initial_triplets, final_triplets, filtered_triplets, ontology_filtered = \
            self.extract_triplets_with_ontology_filtering(text, sample_id=sample_id, source_text_id=source_text_id)
        
        # Step 3: Add to database (parent methods)
        if len(initial_triplets) > 0:
            self.aligner.add_initial_triplets(initial_triplets, sample_id=sample_id)
        if len(final_triplets) > 0:
            self.aligner.add_triplets(final_triplets, sample_id=sample_id)
        if len(filtered_triplets) > 0:
            self.aligner.add_filtered_triplets(filtered_triplets, sample_id=sample_id)
        if len(ontology_filtered) > 0:
            self.aligner.add_ontology_filtered_triplets(ontology_filtered, sample_id=sample_id)
        
        # Step 4: Create passage-entity edges with type information
        all_entities = set()
        entity_types = {}
        
        for triplet in final_triplets:
            subject = triplet.get("subject", "")
            obj = triplet.get("object", "")
            subject_type = triplet.get("subject_type", "")
            object_type = triplet.get("object_type", "")
            
            if subject:
                all_entities.add(subject)
                if subject_type:
                    entity_types[subject] = subject_type
            if obj:
                all_entities.add(obj)
                if object_type:
                    entity_types[obj] = object_type
        
        for entity in all_entities:
            if entity:
                canonical = self._get_canonical_entity(entity)
                entity_type = entity_types.get(entity)
                self.passage_store.add_passage_entity_edge(passage_id, canonical, entity_type)
        
        return initial_triplets, final_triplets, filtered_triplets, ontology_filtered, passage_id
    
    def _get_canonical_entity(self, entity_name: str) -> str:
        """Find canonical name from entity_aliases collection"""
        collection = self.triplets_db.get_collection("entity_aliases")
        result = collection.find_one({"alias": entity_name})
        if result:
            return result.get("label", entity_name)
        
        result = collection.find_one({"label": entity_name})
        if result:
            return entity_name
        
        return entity_name