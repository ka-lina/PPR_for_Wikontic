# src/wikontic_ppr/passage_graph_builder.py

import igraph as ig
from typing import List, Dict, Tuple
from hipporag.utils.misc_utils import compute_mdhash_id


class PassageGraphBuilder:
    """Simplified graph builder that only creates passage-entity connections"""
    
    def __init__(self, triplets_db, passage_store):
        self.db = triplets_db
        self.passage_store = passage_store
        self.graph = None
        self.name_to_idx = {}
        self.passage_node_keys = []
    
    def build(self):
        """Build graph with passage nodes and entity nodes"""
        
        # Get entities
        entity_collection = self.db.get_collection("entity_aliases")
        entities = list(entity_collection.distinct("label"))
        
        # Get passages
        passages = list(self.passage_store.collection.find())
        passage_ids = [p["passage_id"] for p in passages]
        
        print(f"Building graph with {len(entities)} entities and {len(passages)} passages")
        
        # Create nodes
        all_nodes = [{"name": e, "type": "entity"} for e in entities]
        all_nodes.extend([{"name": pid, "type": "passage"} for pid in passage_ids])
        
        self.graph = ig.Graph(directed=False)
        self.graph.add_vertices(len(all_nodes))
        self.graph.vs["name"] = [n["name"] for n in all_nodes]
        self.graph.vs["type"] = [n["type"] for n in all_nodes]
        
        self.name_to_idx = {v["name"]: i for i, v in enumerate(self.graph.vs)}
        self.passage_node_keys = passage_ids
        
        # Add edges from triplets
        triplet_collection = self.db.get_collection("triplets")
        triplets = list(triplet_collection.find())
        
        edges = []
        weights = []
        
        for t in triplets:
            subj = t["subject"]
            obj = t["object"]
            if subj in self.name_to_idx and obj in self.name_to_idx:
                edges.append((self.name_to_idx[subj], self.name_to_idx[obj]))
                weights.append(1.0)
        
        # Add passage-entity edges
        edge_collection = self.passage_store.edge_collection
        for edge in edge_collection.find():
            pid = edge["passage_id"]
            ent = edge["entity_name"]
            if pid in self.name_to_idx and ent in self.name_to_idx:
                edges.append((self.name_to_idx[pid], self.name_to_idx[ent]))
                weights.append(1.0)
        
        print(f"Adding {len(edges)} edges")
        self.graph.add_edges(edges)
        self.graph.es["weight"] = weights
        
        return self.graph