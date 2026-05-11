# src/data/embedding_creator.py

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch

# Add Wikontic and HippoRAG paths
wikontic_path = Path("../Wikontic")
hipporag_path = Path("../HippoRAG")
sys.path.insert(0, str(wikontic_path))
sys.path.insert(0, str(hipporag_path))

# Import from both systems
from wikontic.utils.structured_aligner import Aligner
from hipporag.utils.misc_utils import compute_mdhash_id
from hipporag.utils.embed_utils import retrieve_knn


class EmbeddingCreator:
    def __init__(self, embedding_model_name: str = "facebook/contriever", device: str = "cuda"):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model_name = embedding_model_name
        self.batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", "16"))
        self._load_model()
        
    def _load_model(self):
        """Load embedding model (reuse Wikontic's method)"""
        from transformers import AutoTokenizer, AutoModel

        cache_dir = os.environ.get("HF_HOME")
        local_files_only = os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_safetensors=False,
        ).to(self.device)
        self.model.eval()
        
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self.mean_pooling(outputs[0], inputs["attention_mask"])
        
        return embeddings.cpu().numpy()

    def batch_encode(self, texts: List[str], instruction: str = None, norm: bool = True) -> np.ndarray:
        """
        HippoRAG-compatible batch_encode method.
        
        Args:
            texts: List of strings to encode
            instruction: Optional instruction text (ignored for now)
            norm: Whether to normalize embeddings
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        batches = []
        for start in range(0, len(texts), self.batch_size):
            batches.append(self.encode(texts[start:start + self.batch_size]))

        embeddings = np.vstack(batches)
        
        if norm:
            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def create_entity_embeddings(self, entities: List[str]) -> Dict[str, np.ndarray]:
        """Create embeddings for canonical entities"""
        entity_embeddings = {}
        
        # Batch encode
        embeddings = self.encode(entities)
        
        for entity, embedding in zip(entities, embeddings):
            entity_id = compute_mdhash_id(entity, prefix="entity-")
            entity_embeddings[entity_id] = embedding
            
        return entity_embeddings
    
    def create_fact_embeddings(self, triplets: List[Tuple[str, str, str]]) -> Dict[str, np.ndarray]:
        """Create embeddings for facts (as strings)"""
        fact_strings = [f"({s}, {p}, {o})" for s, p, o in triplets]
        fact_embeddings = {}
        
        embeddings = self.encode(fact_strings)
        
        for fact_str, embedding in zip(fact_strings, embeddings):
            fact_id = compute_mdhash_id(fact_str, prefix="fact-")
            fact_embeddings[fact_id] = embedding
            
        return fact_embeddings
    
    def create_passage_embeddings(self, passages: List[str]) -> Dict[str, np.ndarray]:
        """Create embeddings for passage chunks"""
        passage_embeddings = {}
        
        embeddings = self.encode(passages)
        
        for passage, embedding in zip(passages, embeddings):
            passage_id = compute_mdhash_id(passage, prefix="chunk-")
            passage_embeddings[passage_id] = embedding
            
        return passage_embeddings