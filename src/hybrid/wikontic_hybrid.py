# src/hybrid/wikontic_hybrid.py

from typing import List

from src.hybrid.passage_manager import PassageManager
from src.wikontic_ppr.wikontic_ppr_inference import WikonticPPRInference


class WikonticHybridRetriever(WikonticPPRInference):
    """
    Extends WikonticPPRInference to properly manage passages
    """
    
    def __init__(self, extractor, aligner, triplets_db, ontology_db, 
                 embedding_model, passages_db=None):
        super().__init__(extractor, aligner, triplets_db, embedding_model, passages_db=passages_db)
        self.ontology_db = ontology_db
        
        # Use separate database or same one
        passages_db = passages_db or triplets_db
        self.passage_manager = PassageManager(passages_db, embedding_model)
    
    def process_document_with_passages(self, document: str, doc_id: str):
        """Process a document: add passages, extract triplets, create edges"""
        
        # Step 1: Add passages (chunk the document)
        print(f"  Adding passages for doc {doc_id}...")
        chunk_size = 200  # Characters per passage
        passages = self._chunk_by_sentences(document, chunk_size)
        
        passage_ids = []
        for chunk_idx, passage_text in enumerate(passages):
            passage_id = self.passage_manager.add_passage(
                text=passage_text,
                source_doc_id=doc_id,
                chunk_index=chunk_idx
            )
            passage_ids.append(passage_id)
        
        # Step 2: Extract triplets from the FULL document (not per passage)
        print(f"  Extracting triplets...")
        initial, final, filtered, ontology_filtered = self.extract_triplets_with_ontology_filtering(
            document, sample_id=doc_id, source_text_id=doc_id
        )
        
        # Step 3: Add triplets to database (refined ones go to "triplets" collection)
        if final:
            self.aligner.add_triplets(final, sample_id=doc_id)
        
        # Step 4: Create passage-entity edges based on extracted entities
        print(f"  Creating passage-entity edges...")
        all_entities = set()
        
        for triplet in final:
            subject = triplet.get("subject", "")
            obj = triplet.get("object", "")
            subject_type = triplet.get("subject_type", "")
            object_type = triplet.get("object_type", "")
            
            if subject:
                all_entities.add((subject, subject_type))
            if obj:
                all_entities.add((obj, object_type))
        
        # For each passage, determine which entities appear in it
        for passage_id, passage_text in zip(passage_ids, passages):
            for entity_name, entity_type in all_entities:
                # Simple check: does entity name appear in passage?
                if entity_name.lower() in passage_text.lower():
                    self.passage_manager.add_passage_entity_edge(
                        passage_id=passage_id,
                        entity_name=entity_name,
                        entity_type=entity_type,
                        weight=1.0
                    )
        
        print(f"  Added {len(passage_ids)} passages, {len(final)} triplets")
        
        return passage_ids, final
    
    def _chunk_by_sentences(self, text: str, target_chars: int = 500) -> List[str]:
        """Chunk text by sentences, respecting sentence boundaries"""
        
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            sentence_len = len(sentence)
            
            if current_length + sentence_len > target_chars and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def get_passage_for_retrieval(self, passage_id: str) -> str:
        """Get passage content for retrieval results"""
        passage = self.passage_manager.get_passage(passage_id)
        return passage["content"] if passage else None