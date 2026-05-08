# src/wikontic_ppr/structured_graph_builder.py

import igraph as ig
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm


class StructuredPPRGraphBuilder:
    """Build igraph from Wikontic MongoDB with ontology type hierarchy"""
    
    def __init__(self, triplets_db, ontology_db, passage_store):
        self.db = triplets_db
        self.ontology_db = ontology_db
        self.passage_store = passage_store
        self.graph = None
        self.name_to_idx = {}
        self.passage_node_keys = []
        
    def build(self):
        """Build the full graph for PPR with type hierarchy"""
        print("Collecting entity nodes...")
        entity_nodes = self._get_entity_nodes()
        
        print("Collecting passage nodes...")
        passage_nodes = self._get_passage_nodes()
        
        all_nodes = entity_nodes + passage_nodes
        self.passage_node_keys = [p["name"] for p in passage_nodes]
        
        print(f"Creating graph with {len(all_nodes)} nodes...")
        self.graph = ig.Graph(directed=False)
        self.graph.add_vertices(len(all_nodes))
        self.graph.vs["name"] = [n["name"] for n in all_nodes]
        self.graph.vs["type"] = [n["type"] for n in all_nodes]
        self.graph.vs["entity_type"] = [n.get("entity_type", "") for n in all_nodes]
        
        self.name_to_idx = {v["name"]: i for i, v in enumerate(self.graph.vs)}
        
        edges = []
        weights = []
        
        print("Adding triplet edges with type-based weights...")
        triplet_edges = self._get_weighted_triplet_edges()
        for src, tgt, weight in tqdm(triplet_edges):
            if src in self.name_to_idx and tgt in self.name_to_idx:
                edges.append((self.name_to_idx[src], self.name_to_idx[tgt]))
                weights.append(weight)
        
        print("Adding passage-entity edges...")
        passage_edges = self._get_passage_edges()
        for passage_id, entity_name, weight in tqdm(passage_edges):
            if passage_id in self.name_to_idx and entity_name in self.name_to_idx:
                edges.append((self.name_to_idx[passage_id], self.name_to_idx[entity_name]))
                weights.append(weight)
        
        print("Adding type hierarchy edges...")
        type_edges = self._get_type_hierarchy_edges()
        for parent, child, weight in tqdm(type_edges):
            if parent in self.name_to_idx and child in self.name_to_idx:
                edges.append((self.name_to_idx[parent], self.name_to_idx[child]))
                weights.append(weight)
        
        print(f"Adding {len(edges)} edges...")
        self.graph.add_edges(edges)
        self.graph.es["weight"] = weights
        
        return self.graph
    
    def _get_entity_nodes(self) -> List[Dict]:
        """Get entities with their types from entity_aliases"""
        collection = self.db.get_collection("entity_aliases")
        
        # Get distinct entities with their types
        pipeline = [
            {"$group": {
                "_id": "$label",
                "entity_type": {"$first": "$entity_type"}
            }}
        ]
        results = list(collection.aggregate(pipeline))
        
        nodes = []
        for r in results:
            node = {"name": r["_id"], "type": "entity"}
            if r.get("entity_type"):
                node["entity_type"] = r["entity_type"]
            nodes.append(node)
        
        return nodes
    
    def _get_passage_nodes(self) -> List[Dict]:
        passages = list(self.passage_store.collection.find())
        return [{"name": p["passage_id"], "type": "passage"} for p in passages]
    
    def _get_weighted_triplet_edges(self) -> List[Tuple[str, str, float]]:
        """Get triplet edges with IDF weights using ontology"""
        collection = self.db.get_collection("triplets")
        triplets = list(collection.find())
        
        # Count passages per entity for IDF
        total_passages = self.passage_store.collection.count_documents({})
        entity_passage_count = {}
        
        edge_counts = self.passage_store.edge_collection.aggregate([
            {"$group": {"_id": "$entity_name", "count": {"$sum": 1}}}
        ])
        
        for item in edge_counts:
            entity_passage_count[item["_id"]] = item["count"]
        
        edges = []
        for triplet in triplets:
            subject = triplet.get("subject", "")
            obj = triplet.get("object", "")
            
            if not subject or not obj:
                continue
            
            # IDF weight based on entity frequency
            subj_count = entity_passage_count.get(subject, 1)
            obj_count = entity_passage_count.get(obj, 1)
            weight = np.log(total_passages / (min(subj_count, obj_count) + 1))
            weight = max(0.1, min(weight, 1.0))  # Clamp between 0.1 and 1.0
            
            edges.append((subject, obj, weight))
        
        return edges
    
    def _get_passage_edges(self) -> List[Tuple[str, str, float]]:
        edges = list(self.passage_store.edge_collection.find())
        return [(e["passage_id"], e["entity_name"], e.get("weight", 1.0)) for e in edges]
    
    def _get_type_hierarchy_edges(self) -> List[Tuple[str, str, float]]:
        """Add edges from entities to their parent types"""
        collection = self.ontology_db.get_collection("entity_types")
        
        # Get all entity types and their parents
        type_hierarchy = list(collection.find({}, {"label": 1, "parent_type_ids": 1}))
        
        # Get mapping from type ID to label
        type_id_to_label = {
            t["entity_type_id"]: t["label"] 
            for t in type_hierarchy if "entity_type_id" in t
        }
        
        edges = []
        for entity_type in type_hierarchy:
            if "label" not in entity_type:
                continue
                
            child_label = entity_type["label"]
            parent_ids = entity_type.get("parent_type_ids", [])
            
            for parent_id in parent_ids:
                parent_label = type_id_to_label.get(parent_id)
                if parent_label:
                    # Weight decays with hierarchy depth
                    edges.append((child_label, parent_label, 0.5))
        
        return edges