# src/hybrid/wikontic_hipporag.py

import sys
sys.path.insert(0, "../HippoRAG")

from hipporag.HippoRAG import HippoRAG
from hipporag.utils.misc_utils import compute_mdhash_id
from typing import List, Dict, Optional
import numpy as np

import json
import re
import ast
import numpy as np
from typing import List, Tuple

def parse_fact_string(fact_str: str) -> Tuple[str, str, str]:
    """
    Parse a fact string like '(VIVA Media GmbH, original name, VIVA Media AG)'
    into a tuple of strings.
    """
    # Remove parentheses and whitespace
    fact_str = fact_str.strip()
    if fact_str.startswith('(') and fact_str.endswith(')'):
        fact_str = fact_str[1:-1]
    
    # Split by comma, but don't split inside quotes (there are no quotes)
    # Simple split works for this format
    parts = [p.strip() for p in fact_str.split(',')]
    
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2])
    return None


def normalize_graph_value(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(normalize_graph_value(item) for item in value)
    if value is None:
        return ""
    return str(value)


class WikonticHippoRAG(HippoRAG):
    """
    Inherits from HippoRAG and uses its retrieve function.
    Uses EmbeddingStore's public API to access data.
    """
    
    def __init__(self, triplets_db, embedding_model, global_config=None, **kwargs):
        super().__init__(global_config=global_config, **kwargs)
        
        self.triplets_db = triplets_db
        self.embedding_model = embedding_model
        
        # Collections
        self.triplets_collection = triplets_db.get_collection("triplets")
        self.passages_collection = triplets_db.get_collection("passages")
        self.passage_edges_collection = triplets_db.get_collection("passage_entity_edges")
        self.entity_aliases_collection = triplets_db.get_collection("entity_aliases")
        
    def index_from_wikontic(self, passages: List[str]):
        """
        Build index using Wikontic's refined triplets.
        Uses EmbeddingStore's public methods exclusively.
        """
        
        print("="*60)
        print("Building WikonticHippoRAG Index")
        print("="*60)
        
        # Step 1: Load refined triplets from Wikontic
        print("Loading refined triplets from Wikontic...")
        triplets = list(self.triplets_collection.find())
        print(f"  Loaded {len(triplets)} refined triplets")
        
        # Step 2: Extract entities and facts
        entity_nodes = set()
        facts = []
        
        for triplet in triplets:
            subject = normalize_graph_value(triplet.get("subject", "")).strip()
            relation = normalize_graph_value(triplet.get("relation", "")).strip()
            obj = normalize_graph_value(triplet.get("object", "")).strip()
            
            if subject and obj:
                entity_nodes.add(subject)
                entity_nodes.add(obj)
                facts.append((subject, relation, obj))
        
        print(f"  Extracted {len(entity_nodes)} entities, {len(facts)} facts")
        
        # Step 3: Insert passages into chunk_embedding_store
        print("Inserting passages into chunk_embedding_store...")
        self.chunk_embedding_store.insert_strings(passages)
        
        # Step 4: Insert entities into entity_embedding_store
        print("Inserting entities into entity_embedding_store...")
        entity_list = list(entity_nodes)
        self.entity_embedding_store.insert_strings(entity_list)
        
        # Step 5: Insert facts into fact_embedding_store
        print("Inserting facts into fact_embedding_store...")
        fact_strings = [json.dumps([s, p, o], ensure_ascii=False) for s, p, o in facts]
        self.fact_embedding_store.insert_strings(fact_strings)
        
        # Step 6: Build graph using public API to get hash IDs
        print("Building graph...")
        self._build_graph_from_wikontic(passages, facts, entity_nodes)
        
        # Step 7: Add synonymy edges
        if hasattr(self.global_config, 'add_synonymy_edges') and self.global_config.add_synonymy_edges:
            print("Adding synonymy edges...")
            self.add_synonymy_edges()
        
        # Step 8: Augment and save
        self.augment_graph()
        self.save_igraph()
        
        # Step 9: Prepare for retrieval
        self.prepare_retrieval_objects()
        
        self.ready_to_retrieve = True
        print(f"Index built: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")
        
        return self.graph
    
    def _build_graph_from_wikontic(self, passages: List[str], facts: List[tuple], entity_nodes: set):
        """Build graph using EmbeddingStore's public API"""
        
        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}
        
        # Get mapping from content to hash ID using get_hash_id method
        entity_to_hash = {}
        for entity in entity_nodes:
            try:
                # Try to get existing hash ID
                hash_id = self.entity_embedding_store.get_hash_id(entity)
                entity_to_hash[entity] = hash_id
            except:
                # Create new hash ID if not found
                entity_to_hash[entity] = compute_mdhash_id(entity, prefix="entity-")
        
        # Get passage hash IDs
        passage_to_hash = {}
        for passage in passages:
            try:
                hash_id = self.chunk_embedding_store.get_hash_id(passage)
                passage_to_hash[passage] = hash_id
            except:
                passage_to_hash[passage] = compute_mdhash_id(passage, prefix="chunk-")
        
        # Add fact edges (entity-entity)
        print("  Adding fact edges...")
        for subject, relation, obj in facts:
            subj_hash = entity_to_hash.get(subject)
            obj_hash = entity_to_hash.get(obj)
            
            if subj_hash and obj_hash:
                self.node_to_node_stats[(subj_hash, obj_hash)] = self.node_to_node_stats.get((subj_hash, obj_hash), 0) + 1
                self.node_to_node_stats[(obj_hash, subj_hash)] = self.node_to_node_stats.get((obj_hash, subj_hash), 0) + 1
        
        # Add passage-entity edges using Wikontic's passage_entity_edges
        print("  Adding passage-entity edges...")
        for passage in passages:
            passage_hash = passage_to_hash.get(passage)
            if not passage_hash:
                continue
            
            # Find passage in MongoDB by content
            passage_doc = self.passages_collection.find_one({"content": passage})
            if passage_doc:
                passage_id = passage_doc.get("passage_id")
                edges = self.passage_edges_collection.find({"passage_id": passage_id})
                
                for edge in edges:
                    entity_name = normalize_graph_value(edge.get("entity_name")).strip()
                    entity_hash = entity_to_hash.get(entity_name)
                    
                    if entity_hash:
                        self.node_to_node_stats[(passage_hash, entity_hash)] = 1.0
                        if entity_hash not in self.ent_node_to_chunk_ids:
                            self.ent_node_to_chunk_ids[entity_hash] = set()
                        self.ent_node_to_chunk_ids[entity_hash].add(passage_hash)
        
        print(f"  Added {len(self.node_to_node_stats)} edges")
    
    def retrieve_from_wikontic(self, queries: List[str], num_to_retrieve: int = None):
        """Use HippoRAG's native retrieve function"""
        if not self.ready_to_retrieve:
            raise ValueError("Must call index_from_wikontic() before retrieval")
        
        return super().retrieve(queries, num_to_retrieve)

    # def rerank_facts(self, query: str, query_fact_scores: np.ndarray):
    #     """Override with safe fact parsing and proper entity mapping"""
    #     link_top_k = self.global_config.linking_top_k
        
    #     if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
    #         return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
        
    #     # Get top candidate facts
    #     candidate_count = min(link_top_k * 2, len(query_fact_scores))
    #     candidate_indices = np.argsort(query_fact_scores)[-candidate_count:][::-1].tolist()
        
    #     # Get fact content with safe parsing
    #     fact_ids = [self.fact_node_keys[idx] for idx in candidate_indices]
    #     candidate_facts = []
    #     valid_indices = []
        
    #     for idx, fid in zip(candidate_indices, fact_ids):
    #         row = self.fact_embedding_store.get_row(fid)
    #         content = row['content']
            
    #         # Parse the fact string
    #         fact_tuple = self._parse_fact_string(content)
    #         if fact_tuple:
    #             subject, predicate, obj = fact_tuple
    #             # Verify that entities exist in the graph
    #             subj_exists = self._entity_exists_in_graph(subject)
    #             obj_exists = self._entity_exists_in_graph(obj)
                
    #             if subj_exists and obj_exists:
    #                 candidate_facts.append(fact_tuple)
    #                 valid_indices.append(idx)
    #             else:
    #                 print(f"  Entity not in graph: subject={subj_exists}, object={obj_exists} for {fact_tuple}")
        
    #     if not candidate_facts:
    #         print("  No valid facts found, returning empty")
    #         return [], [], {'error': 'No valid facts'}
        
    #     # Keep only top_k valid facts
    #     valid_indices = valid_indices[:link_top_k]
    #     candidate_facts = candidate_facts[:link_top_k]
        
    #     return valid_indices, candidate_facts, {'skip_rerank': True}
    
    # def _parse_fact_string(self, fact_str: str) -> tuple:
    #     """Parse fact string like '(subject, predicate, object)'"""
    #     try:
    #         # Remove parentheses
    #         fact_str = fact_str.strip()
    #         if fact_str.startswith('(') and fact_str.endswith(')'):
    #             fact_str = fact_str[1:-1]
            
    #         # Split by comma (simple approach for unquoted strings)
    #         parts = [p.strip() for p in fact_str.split(',')]
            
    #         if len(parts) >= 3:
    #             return (parts[0], parts[1], parts[2])
    #     except Exception as e:
    #         print(f"  Parse error: {e}")
        
    #     return None
    
    def _entity_exists_in_graph(self, entity_name: str) -> bool:
        """Check if entity exists in the graph"""
        try:
            # Try to get hash ID from entity embedding store
            hash_id = self.entity_embedding_store.get_hash_id(entity_name)
            # Check if it's in the graph nodes
            return hash_id in self.name_to_idx if hasattr(self, 'name_to_idx') else True
        except:
            return False

    # def _build_graph_from_wikontic(self, passages: List[str], facts: List[tuple], entity_nodes: set):
    #     """Build graph using direct access to triplets collection"""
        
    #     self.node_to_node_stats = {}
    #     self.ent_node_to_chunk_ids = {}
        
    #     # Get entity hashes
    #     entity_to_hash = {}
    #     for entity in entity_nodes:
    #         try:
    #             hash_id = self.entity_embedding_store.get_hash_id(entity)
    #             entity_to_hash[entity] = hash_id
    #         except:
    #             entity_to_hash[entity] = compute_mdhash_id(entity, prefix="entity-")
        
    #     # Get passage hashes
    #     passage_to_hash = {}
    #     for passage in passages:
    #         try:
    #             hash_id = self.chunk_embedding_store.get_hash_id(passage)
    #             passage_to_hash[passage] = hash_id
    #         except:
    #             passage_to_hash[passage] = compute_mdhash_id(passage, prefix="chunk-")
        
    #     # Add fact edges - read directly from triplets collection (no parsing!)
    #     print("  Adding fact edges from triplets collection...")
    #     triplets = list(self.triplets_collection.find())
        
    #     for triplet in triplets:
    #         subject = triplet.get("subject", "")
    #         obj = triplet.get("object", "")
            
    #         if subject and obj:
    #             subj_hash = entity_to_hash.get(subject)
    #             obj_hash = entity_to_hash.get(obj)
                
    #             if subj_hash and obj_hash:
    #                 self.node_to_node_stats[(subj_hash, obj_hash)] = self.node_to_node_stats.get((subj_hash, obj_hash), 0) + 1
    #                 self.node_to_node_stats[(obj_hash, subj_hash)] = self.node_to_node_stats.get((obj_hash, subj_hash), 0) + 1
        
    #     print(f"  Added {len(self.node_to_node_stats)} entity-entity edges")
        
    #     # Add passage-entity edges
    #     print("  Adding passage-entity edges...")
    #     for passage in passages:
    #         passage_hash = passage_to_hash.get(passage)
    #         if not passage_hash:
    #             continue
            
    #         passage_doc = self.passages_collection.find_one({"content": passage})
    #         if passage_doc:
    #             passage_id = passage_doc.get("passage_id")
    #             edges = self.passage_edges_collection.find({"passage_id": passage_id})
                
    #             for edge in edges:
    #                 entity_name = edge.get("entity_name")
    #                 entity_hash = entity_to_hash.get(entity_name)
                    
    #                 if entity_hash:
    #                     self.node_to_node_stats[(passage_hash, entity_hash)] = 1.0
    #                     if entity_hash not in self.ent_node_to_chunk_ids:
    #                         self.ent_node_to_chunk_ids[entity_hash] = set()
    #                     self.ent_node_to_chunk_ids[entity_hash].add(passage_hash)
        
    #     print(f"  Total edges: {len(self.node_to_node_stats)}")
    
    def _rerank_with_llm(self, query: str, candidate_facts: List[Tuple]) -> List[Tuple]:
        """Rerank facts using LLM with robust parsing"""
        
        # Format facts for prompt
        facts_str = json.dumps([list(f) for f in candidate_facts], indent=2)
        
        prompt = f"""Given the question, select which facts are relevant.

Question: {query}

Facts:
{facts_str}

Return ONLY the relevant facts as a JSON list of lists. Example: [["subject", "predicate", "object"]]

Relevant facts:"""

        response = self.llm_model.infer(prompt)
        
        # Robust parsing
        return self._parse_fact_response(response, candidate_facts)
    
    def _parse_fact_response(self, response: str, candidate_facts: List[Tuple]) -> List[Tuple]:
        """Parse LLM response to extract facts"""
        
        # Try different parsing strategies
        parsed = None
        
        # Strategy 1: Find JSON in response
        json_pattern = r'\[.*\]'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except:
                pass
        
        # Strategy 2: Try ast.literal_eval
        if parsed is None:
            try:
                parsed = ast.literal_eval(response)
            except:
                pass
        
        # Strategy 3: Extract by matching with candidates
        if parsed is None or not isinstance(parsed, list):
            # Return all candidates (no filtering)
            return candidate_facts
        
        # Convert to tuple format
        result = []
        for item in parsed:
            if isinstance(item, list) and len(item) >= 3:
                result.append(tuple(item[:3]))
            elif isinstance(item, dict):
                result.append((item.get('subject', ''), item.get('predicate', ''), item.get('object', '')))
        
        return result if result else candidate_facts
    
    def _facts_match(self, fact1: Tuple, fact2: Tuple) -> bool:
        """Check if two facts are the same (fuzzy matching)"""
        return all(str(f1).lower().strip() == str(f2).lower().strip() 
                   for f1, f2 in zip(fact1, fact2))