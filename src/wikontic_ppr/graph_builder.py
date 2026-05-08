# src/wikontic_ppr/graph_builder.py

import igraph as ig
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import pickle


class PPRGraphBuilder:
    def __init__(self, triplets_db, passage_store, cache_dir: str = "./cache"):
        self.db = triplets_db
        self.passage_store = passage_store
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.graph = None
        self.name_to_idx = {}
        self.passage_node_keys = []
    
    def get_graph_path(self) -> Path:
        """Get path for saved graph file"""
        return self.cache_dir / "wikontic_ppr_graph.pickle"
    
    def get_metadata_path(self) -> Path:
        """Get path for metadata file"""
        return self.cache_dir / "wikontic_ppr_metadata.pickle"
    
    def save(self):
        """Save graph and metadata to disk"""
        if self.graph is None:
            raise ValueError("No graph to save")
        
        print(f"Saving graph to {self.get_graph_path()}...")
        self.graph.write_pickle(str(self.get_graph_path()))
        
        # Save metadata
        metadata = {
            'name_to_idx': self.name_to_idx,
            'passage_node_keys': self.passage_node_keys,
            'num_nodes': self.graph.vcount(),
            'num_edges': self.graph.ecount()
        }
        
        with open(self.get_metadata_path(), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Graph saved: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")
    
    def load(self) -> bool:
        """Load graph and metadata from disk if exists"""
        graph_path = self.get_graph_path()
        metadata_path = self.get_metadata_path()
        
        if not graph_path.exists() or not metadata_path.exists():
            return False
        
        print(f"Loading graph from {graph_path}...")
        self.graph = ig.Graph.Read_Pickle(str(graph_path))
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.name_to_idx = metadata['name_to_idx']
        self.passage_node_keys = metadata['passage_node_keys']
        
        print(f"Graph loaded: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")
        return True


    def build(self, force_rebuild: bool = False):
        """Build graph, loading from cache if available"""
        
        if not force_rebuild and self.load():
            return self.graph
        # Collect all nodes
        entity_nodes = self._get_entity_nodes()
        passage_nodes = self._get_passage_nodes()
        
        all_nodes = entity_nodes + passage_nodes
        self.passage_node_keys = [p["name"] for p in passage_nodes]
        
        # Create graph
        self.graph = ig.Graph(directed=False)
        self.graph.add_vertices(len(all_nodes))
        self.graph.vs["name"] = [n["name"] for n in all_nodes]
        self.graph.vs["type"] = [n["type"] for n in all_nodes]
        
        self.name_to_idx = {v["name"]: i for i, v in enumerate(self.graph.vs)}
        
        # Add edges
        edges = []
        weights = []
        
        # Entity-entity edges (from triplets)
        triplet_edges = self._get_triplet_edges()
        for src, tgt, weight in triplet_edges:
            edges.append((self.name_to_idx[src], self.name_to_idx[tgt]))
            weights.append(weight)
        
        # Passage-entity edges
        passage_edges = self._get_passage_edges()
        for passage_id, entity_name, weight in passage_edges:
            edges.append((self.name_to_idx[passage_id], self.name_to_idx[entity_name]))
            weights.append(weight)
        
        self.graph.add_edges(edges)
        self.graph.es["weight"] = weights

        self.save()
        
        return self.graph
    
    def _get_entity_nodes(self) -> List[Dict]:
        """Get all canonical entities from entity_aliases"""
        collection = self.db.get_collection("entity_aliases")
        distinct_entities = collection.distinct("label")
        
        return [{"name": e, "type": "entity"} for e in distinct_entities]

    
    def _get_passage_nodes(self) -> List[Dict]:
        """Get all passages from passage store"""
        passages = list(self.passage_store.collection.find())
        return [{"name": p["passage_id"], "type": "passage"} for p in passages]
    
    def _get_triplet_edges(self) -> List[Tuple[str, str, float]]:
        """Get entity-entity edges from triplets with IDF weights"""
        collection = self.db.get_collection("triplets")
        triplets = list(collection.find())
        
        edges = []
        for triplet in triplets:
            # Simple weight = 1.0 (no IDF without passage counts)
            edges.append((triplet["subject"], triplet["object"], 1.0))
            edges.append((triplet["object"], triplet["subject"], 1.0))
        
        return edges
    
    def _get_passage_edges(self) -> List[Tuple[str, str, float]]:
        """Get passage-entity edges"""
        edges = list(self.passage_store.edge_collection.find())
        return [(e["passage_id"], e["entity_name"], e.get("weight", 1.0)) for e in edges]