# src/wikontic_ppr/ppr_retriever.py

import numpy as np
from typing import List, Tuple, Optional


class PPRRetriever:
    def __init__(self, graph, passage_node_keys, name_to_idx, embedding_model=None, 
        passage_store=None, extractor=None):
        self.graph = graph
        self.passage_node_keys = passage_node_keys
        self.passage_node_idxs = [name_to_idx[k] for k in passage_node_keys]
        self.name_to_idx = name_to_idx
        self.embedding_model = embedding_model
        self.passage_store = passage_store
        self.extractor = extractor  # Added for answer generation
    
    def retrieve(self, query_entities: List[str], query_text: Optional[str] = None, 
                 damping: float = 0.5, top_k: int = 20) -> Tuple[List[str], List[float]]:
        """Retrieve passages using PPR"""
        
        # Initialize reset distribution
        reset_prob = np.zeros(len(self.graph.vs))
        
        # Weight entities
        for entity in query_entities:
            if entity in self.name_to_idx:
                idx = self.name_to_idx[entity]
                reset_prob[idx] = 1.0 / len(query_entities)
        
        # Fallback to dense retrieval if no entities
        if np.sum(reset_prob) == 0 and query_text and self.embedding_model:
            return self._dense_retrieval(query_text, top_k)
        
        # Run PPR
        ppr_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.graph.vs)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
        
        # Extract passage scores
        passage_scores = [(self.passage_node_keys[i], ppr_scores[idx]) 
                          for i, idx in enumerate(self.passage_node_idxs)]
        
        passage_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [p[0] for p in passage_scores[:top_k]], [p[1] for p in passage_scores[:top_k]]
    
    def _dense_retrieval(self, query_text: str, top_k: int) -> Tuple[List[str], List[float]]:
        """Fallback: dense passage retrieval"""
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        # Search passages by embedding similarity
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "passage_embeddings",
                    "queryVector": query_embedding.tolist(),
                    "path": "embedding",
                    "numCandidates": 150,
                    "limit": top_k
                }
            },
            {"$project": {"passage_id": 1, "score": {"$meta": "vectorSearchScore"}}}
        ]
        
        results = list(self.passage_store.collection.aggregate(pipeline))
        return [r["passage_id"] for r in results], [r["score"] for r in results]

    def answer(self, query: str, passage_ids: List[str], top_k: int = 5) -> str:
        """Generate answer from retrieved passages"""
        if not self.extractor:
            return "No extractor available for answer generation"
        
        # Get passage contents
        passages = []
        for pid in passage_ids[:top_k]:
            passage = self.passage_store.collection.find_one({"passage_id": pid})
            if passage:
                passages.append(passage["content"])
        
        if not passages:
            return "No relevant passages found"
        
        context = "\n\n---\n\n".join(passages)
        prompt = f"""Answer the question based ONLY on the provided passages.

Passages:
{context}

Question: {query}

Answer:"""
        
        return self.extractor.generate(prompt)
    
    def retrieve_and_answer(self, query: str, query_entities: List[str],
                           damping: float = 0.5, top_k: int = 10) -> dict:
        """Convenience method: retrieve then answer"""
        passage_ids, scores = self.retrieve(query_entities, query, damping, top_k)
        answer = self.answer(query, passage_ids)
        
        return {
            'passage_ids': passage_ids,
            'scores': scores,
            'answer': answer
        }