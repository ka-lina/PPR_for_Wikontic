# src/hybrid/wikontic_hipporag.py

import numpy as np
import igraph as ig
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from hipporag.utils.misc_utils import compute_mdhash_id
from hipporag.utils.embed_utils import retrieve_knn


class WikonticHippoRAG:
    """
    Builds HippoRAG-style graph from Wikontic's existing database:
    - Reads refined triplets from MongoDB
    - Reads passages and passage-entity edges from MongoDB
    - Builds graph with PPR support
    """
    
    def __init__(self, triplets_db, embedding_model, config=None):
        self.triplets_db = triplets_db
        self.embedding_model = embedding_model
        
        # Collections
        self.triplets_collection = triplets_db.get_collection("triplets")
        self.passages_collection = triplets_db.get_collection("passages")
        self.passage_edges_collection = triplets_db.get_collection("passage_entity_edges")
        self.entity_aliases_collection = triplets_db.get_collection("entity_aliases")
        
        # Graph structures
        self.graph = None
        self.name_to_idx = {}
        self.passage_node_keys = []
        self.entity_node_keys = []
        self.node_to_node_stats = {}
        self.ent_node_to_passage_ids = {}
        
        # Embeddings (will be loaded)
        self.passage_embeddings = None
        self.entity_embeddings = None
        self.fact_embeddings = None
        
        self.ready = False
        
    def build_from_database(self, force_rebuild: bool = False):
        """Build graph from existing database collections"""
        
        print("="*60)
        print("Building WikonticHippoRAG Graph")
        print("="*60)
        
        # Step 1: Load all data from MongoDB
        self._load_entities()
        self._load_passages()
        self._load_triplets()
        self._load_passage_edges()
        
        # Step 2: Create embeddings for entities and facts
        self._create_embeddings()
        
        # Step 3: Build graph structure
        self._build_graph()
        
        # Step 4: Add synonymy edges (optional)
        if self.config.get('add_synonymy_edges', True):
            self._add_synonymy_edges()
        
        # Step 5: Augment graph with nodes and edges
        self._augment_graph()
        
        self.ready = True
        print(f"Graph built: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")
        
        return self.graph
    
    def _load_entities(self):
        """Load entities from entity_aliases collection"""
        print("Loading entities...")
        
        # Get distinct canonical entities
        entities = self.entity_aliases_collection.distinct("label")
        self.entity_node_keys = []
        
        for entity in entities:
            if entity and entity.strip():
                entity_hash = compute_mdhash_id(entity, prefix="entity-")
                self.entity_node_keys.append({
                    "name": entity,
                    "hash": entity_hash,
                    "type": "entity"
                })
        
        print(f"  Loaded {len(self.entity_node_keys)} entities")
    
    def _load_passages(self):
        """Load passages from passages collection"""
        print("Loading passages...")
        
        passages = list(self.passages_collection.find())
        self.passage_node_keys = []
        
        for passage in passages:
            passage_hash = passage["passage_id"]
            self.passage_node_keys.append({
                "name": passage_hash,
                "content": passage.get("content", ""),
                "type": "passage"
            })
        
        print(f"  Loaded {len(self.passage_node_keys)} passages")
    
    def _load_triplets(self):
        """Load refined triplets from triplets collection"""
        print("Loading refined triplets...")
        
        triplets = list(self.triplets_collection.find())
        self.facts = []
        
        # Create mapping from entity name to hash
        self.entity_name_to_hash = {}
        for entity in self.entity_node_keys:
            self.entity_name_to_hash[entity["name"]] = entity["hash"]
        
        for triplet in triplets:
            subject = triplet.get("subject", "")
            relation = triplet.get("relation", "")
            obj = triplet.get("object", "")
            
            if subject and obj:
                subj_hash = self.entity_name_to_hash.get(subject)
                obj_hash = self.entity_name_to_hash.get(obj)
                
                if subj_hash and obj_hash:
                    self.facts.append({
                        "subject": subject,
                        "subject_hash": subj_hash,
                        "relation": relation,
                        "object": obj,
                        "object_hash": obj_hash,
                        "fact_string": f"({subject}, {relation}, {obj})"
                    })
        
        print(f"  Loaded {len(self.facts)} refined triplets")
    
    def _load_passage_edges(self):
        """Load passage-entity edges"""
        print("Loading passage-entity edges...")
        
        edges = list(self.passage_edges_collection.find())
        self.passage_edges = []
        
        # Track which passages contain which entities (for IDF weighting)
        self.entity_passage_count = {}
        
        for edge in edges:
            passage_id = edge.get("passage_id")
            entity_name = edge.get("entity_name")
            
            if passage_id and entity_name:
                self.passage_edges.append({
                    "passage_id": passage_id,
                    "entity_name": entity_name
                })
                
                # Count passages per entity
                self.entity_passage_count[entity_name] = self.entity_passage_count.get(entity_name, 0) + 1
        
        print(f"  Loaded {len(self.passage_edges)} passage-entity edges")
    
    def _create_embeddings(self):
        """Create embeddings for passages, entities, and facts"""
        
        # Passage embeddings
        print("Creating passage embeddings...")
        passage_texts = [p["content"] for p in self.passage_node_keys if p.get("content")]
        if passage_texts:
            self.passage_embeddings = self.embedding_model.encode(passage_texts)
        
        # Entity embeddings
        print("Creating entity embeddings...")
        entity_names = [e["name"] for e in self.entity_node_keys]
        if entity_names:
            self.entity_embeddings = self.embedding_model.encode(entity_names)
        
        # Fact embeddings
        print("Creating fact embeddings...")
        fact_strings = [f["fact_string"] for f in self.facts]
        if fact_strings:
            self.fact_embeddings = self.embedding_model.encode(fact_strings)
    
    def _build_graph(self):
        """Build igraph from loaded data"""
        
        print("Building graph structure...")
        
        # Combine all nodes
        all_nodes = []
        for entity in self.entity_node_keys:
            all_nodes.append({
                "name": entity["hash"],
                "display_name": entity["name"],
                "type": "entity"
            })
        
        for passage in self.passage_node_keys:
            all_nodes.append({
                "name": passage["name"],
                "display_name": passage["name"],
                "type": "passage"
            })
        
        # Create graph
        self.graph = ig.Graph(directed=False)
        self.graph.add_vertices(len(all_nodes))
        self.graph.vs["name"] = [n["name"] for n in all_nodes]
        self.graph.vs["display_name"] = [n.get("display_name", n["name"]) for n in all_nodes]
        self.graph.vs["type"] = [n["type"] for n in all_nodes]
        
        # Create name to index mapping
        self.name_to_idx = {n["name"]: i for i, n in enumerate(all_nodes)}
        
        # Add edges
        edges = []
        weights = []
        
        # Add fact edges (entity-entity)
        print("  Adding fact edges...")
        total_passages = len(self.passage_node_keys)
        
        for fact in tqdm(self.facts):
            subj_hash = fact["subject_hash"]
            obj_hash = fact["object_hash"]
            
            if subj_hash in self.name_to_idx and obj_hash in self.name_to_idx:
                # IDF weight based on entity frequency
                subj_count = self.entity_passage_count.get(fact["subject"], 1)
                obj_count = self.entity_passage_count.get(fact["object"], 1)
                weight = np.log(total_passages / (min(subj_count, obj_count) + 1))
                weight = max(0.1, min(weight, 1.0))
                
                edges.append((self.name_to_idx[subj_hash], self.name_to_idx[obj_hash]))
                weights.append(weight)
        
        # Add passage-entity edges
        print("  Adding passage-entity edges...")
        for edge in tqdm(self.passage_edges):
            passage_id = edge["passage_id"]
            entity_name = edge["entity_name"]
            entity_hash = self.entity_name_to_hash.get(entity_name)
            
            if passage_id in self.name_to_idx and entity_hash in self.name_to_idx:
                edges.append((self.name_to_idx[passage_id], self.name_to_idx[entity_hash]))
                weights.append(1.0)
                
                # Track entity -> passage mapping for later
                if entity_hash not in self.ent_node_to_passage_ids:
                    self.ent_node_to_passage_ids[entity_hash] = set()
                self.ent_node_to_passage_ids[entity_hash].add(passage_id)
        
        # Add all edges
        self.graph.add_edges(edges)
        self.graph.es["weight"] = weights
        
        print(f"  Added {len(edges)} edges")
    
    def _add_synonymy_edges(self, threshold: float = 0.7, top_k: int = 10):
        """Add synonymy edges between similar entities (HippoRAG style)"""
        
        if len(self.entity_node_keys) < 2:
            return
        
        print("Adding synonymy edges...")
        
        entity_names = [e["name"] for e in self.entity_node_keys]
        entity_hashes = [e["hash"] for e in self.entity_node_keys]
        
        # Get embeddings for entities
        embeddings = self.entity_embeddings
        
        # Find similar entities
        similar_pairs = retrieve_knn(
            query_ids=entity_hashes,
            key_ids=entity_hashes,
            query_vecs=embeddings,
            key_vecs=embeddings,
            k=top_k
        )
        
        synonym_edges = []
        synonym_weights = []
        
        for entity_hash, (similar_hashes, scores) in similar_pairs.items():
            for sim_hash, score in zip(similar_hashes, scores):
                if score >= threshold and entity_hash != sim_hash:
                    if entity_hash in self.name_to_idx and sim_hash in self.name_to_idx:
                        synonym_edges.append((self.name_to_idx[entity_hash], self.name_to_idx[sim_hash]))
                        synonym_weights.append(score * 0.3)  # Lower weight than fact edges
        
        if synonym_edges:
            self.graph.add_edges(synonym_edges)
            self.graph.es[len(self.graph.es) - len(synonym_edges):]["weight"] = synonym_weights
            print(f"  Added {len(synonym_edges)} synonymy edges")
    
    def _augment_graph(self):
        """Finalize graph (add any remaining nodes/edges)"""
        # This is a placeholder - add any post-processing here
        pass
    
    def get_query_embedding(self, query: str, mode: str = "fact") -> np.ndarray:
        """Get query embedding for retrieval"""
        # You'll need to implement instruction-based embedding
        # For now, simple encoding
        return self.embedding_model.encode([query])[0]
    
    def retrieve(self, query: str, query_entities: List[str], 
                 damping: float = 0.5, top_k: int = 20) -> Tuple[List[str], List[float]]:
        """Retrieve passages using PPR"""
        
        if not self.ready:
            raise ValueError("Graph not built. Call build_from_database() first.")
        
        # Initialize reset distribution
        reset_prob = np.zeros(len(self.graph.vs))
        
        # Weight entities
        entity_hashes = []
        for entity_name in query_entities:
            entity_hash = self.entity_name_to_hash.get(entity_name)
            if entity_hash and entity_hash in self.name_to_idx:
                entity_hashes.append(entity_hash)
        
        if entity_hashes:
            for entity_hash in entity_hashes:
                idx = self.name_to_idx[entity_hash]
                # Weight by inverse passage frequency (rare entities get higher weight)
                passage_count = self.entity_passage_count.get(
                    self._get_entity_name_from_hash(entity_hash), 1
                )
                weight = 1.0 / passage_count
                reset_prob[idx] = weight
            
            # Normalize
            if np.sum(reset_prob) > 0:
                reset_prob = reset_prob / np.sum(reset_prob)
        else:
            # Fallback to dense retrieval
            return self._dense_retrieval(query, top_k)
        
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
        passage_scores = []
        for passage in self.passage_node_keys:
            passage_name = passage["name"]
            if passage_name in self.name_to_idx:
                idx = self.name_to_idx[passage_name]
                score = ppr_scores[idx]
                passage_scores.append((passage_name, score))
        
        passage_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [p[0] for p in passage_scores[:top_k]], [p[1] for p in passage_scores[:top_k]]
    
    def _dense_retrieval(self, query: str, top_k: int) -> Tuple[List[str], List[float]]:
        """Fallback dense passage retrieval"""
        query_emb = self.get_query_embedding(query, mode="passage")
        
        scores = []
        for passage in self.passage_node_keys:
            passage_emb = self._get_passage_embedding(passage["name"])
            if passage_emb is not None:
                score = np.dot(query_emb, passage_emb)
                scores.append((passage["name"], score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:top_k]], [s[1] for s in scores[:top_k]]
    
    def _get_entity_name_from_hash(self, entity_hash: str) -> str:
        """Get entity name from hash"""
        for entity in self.entity_node_keys:
            if entity["hash"] == entity_hash:
                return entity["name"]
        return ""
    
    def _get_passage_embedding(self, passage_id: str) -> Optional[np.ndarray]:
        """Get passage embedding from store"""
        passage = self.passages_collection.find_one({"passage_id": passage_id})
        if passage and "embedding" in passage:
            return np.array(passage["embedding"])
        return None