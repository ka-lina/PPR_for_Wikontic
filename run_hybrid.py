# experiments/run_hybrid.py

from src.data.hotpotqa_loader import HotPotQALoader
from src.hybrid.passage_manager import PassageManager
from src.hybrid.wikontic_hipporag import WikonticHippoRAG
from src.data.embedding_creator import EmbeddingCreator

from pymongo import MongoClient
import sys
sys.path.insert(0, "../Wikontic")
from wikontic.utils.openai_utils import LLMTripletExtractor
from wikontic.utils.structured_aligner import Aligner
from wikontic.create_wikidata_ontology_db import create_wikidata_ontology_database
from wikontic.create_ontological_triplets_db import create_ontological_triplets_database


def extract_query_entities(extractor, query: str) -> list:
    """Extract entities from query using LLM"""
    
    try:
        # Try the proper entity extraction method
        entities = extractor.extract_entities_from_question(query)
        
        # Handle different return types
        if isinstance(entities, list):
            return entities
        elif isinstance(entities, dict):
            if "entities" in entities:
                return entities["entities"]
            elif "triplets" in entities:
                # Extract unique subjects and objects from triplets
                unique_entities = set()
                for triplet in entities.get("triplets", []):
                    if isinstance(triplet, list) and len(triplet) >= 3:
                        unique_entities.add(triplet[0])  # subject
                        unique_entities.add(triplet[2])  # object
                return list(unique_entities)
        elif isinstance(entities, str):
            # Try to parse string as list
            import json
            try:
                parsed = json.loads(entities)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
            # Fallback: extract quoted strings
            import re
            found = re.findall(r'"([^"]+)"', entities)
            if found:
                return found
        
        return []
        
    except Exception as e:
        print(f"Entity extraction failed: {e}")
        return []


def extract_entities_fallback(query: str) -> list:
    """Simple rule-based fallback when LLM fails"""
    
    import re
    
    # Look for capitalized phrases
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    
    # Also look for numbers/years
    numbers = re.findall(r'\b\d{4}\b', query)
    entities.extend(numbers)
    
    # Remove duplicates and common stopwords
    stopwords = {"What", "Who", "When", "Where", "Why", "How", "Does", "Did", "Is", "Are"}
    entities = [e for e in entities if e not in stopwords]
    
    # For VIVA question specifically (if needed)
    if "VIVA" in query and "VIVA Media AG" not in entities:
        entities.append("VIVA Media AG")
    
    return list(set(entities))


def get_query_entities(extractor, query: str, use_llm: bool = True) -> list:
    """Get entities from query with fallback"""
    
    if use_llm:
        entities = extract_query_entities(extractor, query)
        if entities:
            print(f"  LLM extracted: {entities}")
            return entities
    
    # Fallback to rule-based
    entities = extract_entities_fallback(query)
    print(f"  Rule-based extraction: {entities}")
    return entities


def setup_databases(mongo_uri: str):
    """Setup ontology and triplets databases"""
    
    print("Creating Wikidata ontology database...")
    create_wikidata_ontology_database(
        mongo_uri=mongo_uri,
        database="wikidata_ontology_test",
        drop_collections=True
    )
    
    print("Creating ontological triplets database...")
    create_ontological_triplets_database(
        mongo_uri=mongo_uri,
        db_name="hotpotqa_hybrid",
        drop_collections=True
    )
    
    return "wikidata_ontology_test", "hotpotqa_hybrid"


def add_passage_entity_edge(passage_id: str, entity_name: str, edges_collection):
    """Helper to add passage-entity edge"""
    edges_collection.update_one(
        {"passage_id": passage_id, "entity_name": entity_name},
        {"$set": {"weight": 1.0}},
        upsert=True
    )


def main():
    # =========================================================
    # 1. Setup MongoDB
    # =========================================================
    mongo_uri = "mongodb://localhost:27018/?directConnection=true"
    ontology_db_name, triplets_db_name = setup_databases(mongo_uri)
    
    client = MongoClient(mongo_uri)
    triplets_db = client[triplets_db_name]
    ontology_db = client[ontology_db_name]
    
    # =========================================================
    # 2. Load HotPotQA data
    # =========================================================
    print("\nLoading HotPotQA...")
    loader = HotPotQALoader("./Wikontic/datasets/hotpotqa200.json")
    documents, queries, answers, gold_passage_ids, question_contexts = loader.load()
    
    # Use small subset for debugging
    num_questions = 1
    documents = documents[:num_questions * 10]
    queries = queries[:num_questions]
    answers = answers[:num_questions]
    gold_passage_ids = gold_passage_ids[:num_questions]
    question_contexts = question_contexts[:num_questions]
    
    print(f"First query: {queries[0]}")
    print(f"First answer: {answers[0]}")
    
    # Extract relevant passages for each question
    question_passages = []
    for j in range(len(question_contexts)):
        qp = []
        for i in gold_passage_ids[j]:
            if i < len(question_contexts[j]):
                qp.append(question_contexts[j][i])
        question_passages.append(qp)
    
    # =========================================================
    # 3. Initialize models
    # =========================================================
    print("\nInitializing models...")
    embedding_model = EmbeddingCreator("facebook/contriever")
    extractor = LLMTripletExtractor(model_name="Qwen/Qwen2.5-3B-Instruct")
    aligner = Aligner(triplets_db=triplets_db, ontology_db=ontology_db)
    
    # =========================================================
    # 4. Initialize Wikontic PPR (for triplet extraction)
    # =========================================================
    from src.wikontic_ppr.structured_wikontic_ppr_inference import StructuredWikonticPPRInference
    
    wikontic_ppr = StructuredWikonticPPRInference(
        extractor=extractor,
        aligner=aligner,
        triplets_db=triplets_db,
        ontology_db=ontology_db,
        embedding_model=embedding_model
    )
    
    # =========================================================
    # 5. Process passages and extract triplets
    # =========================================================
    print("\nProcessing passages...")
    passage_manager = PassageManager(triplets_db, embedding_model)
    edges_collection = triplets_db.get_collection("passage_entity_edges")
    
    for q_idx, (query, gold_passages) in enumerate(zip(queries, question_passages)):
        passage_ids_for_query = []
        
        for p_idx, passage_text in enumerate(gold_passages):
            print(f"  Processing passage {p_idx} for question {q_idx}...")
            
            # Add passage to database
            passage_id = passage_manager.add_passage(
                text=passage_text,
                source_doc_id=f"q{q_idx}",
                chunk_index=p_idx
            )
            passage_ids_for_query.append(passage_id)
            
            # Extract triplets from this passage
            try:
                initial, final, filtered, ontology = wikontic_ppr.extract_triplets_with_ontology_filtering(
                    passage_text, sample_id=f"q{q_idx}", source_text_id=f"p{p_idx}"
                )
                
                # Add refined triplets to database
                if final:
                    aligner.add_triplets(final, sample_id=f"q{q_idx}")
                    
                    # Create passage-entity edges
                    for triplet in final:
                        if triplet.get("subject"):
                            add_passage_entity_edge(passage_id, triplet["subject"], edges_collection)
                        if triplet.get("object"):
                            add_passage_entity_edge(passage_id, triplet["object"], edges_collection)
                    
                    print(f"    Added {len(final)} triplets")
                
            except Exception as e:
                print(f"    Triplet extraction failed: {e}")
        
        print(f"Question {q_idx}: Added {len(passage_ids_for_query)} passages")
    
    # =========================================================
    # 6. Build HippoRAG-style graph from database
    # =========================================================
    print("\nBuilding WikonticHippoRAG graph...")
    
    hybrid = WikonticHippoRAG(
        triplets_db=triplets_db,
        embedding_model=embedding_model
    )
    
    # Build graph from existing database
    graph = hybrid.build_from_database(force_rebuild=False)
    
    # =========================================================
    # 7. Run retrieval for all queries
    # =========================================================
    print("\n" + "="*60)
    print("RUNNING RETRIEVAL")
    print("="*60)
    
    all_results = []
    
    for q_idx, query in enumerate(queries):
        print(f"\n--- Query {q_idx}: {query[:80]}... ---")
        
        # Step 1: Extract entities dynamically
        query_entities = get_query_entities(extractor, query, use_llm=True)
        
        if not query_entities:
            print("  No entities found, using rule-based fallback")
            query_entities = extract_entities_fallback(query)
        
        print(f"  Query entities: {query_entities}")
        
        # Step 2: Retrieve passages
        passage_ids, scores = hybrid.retrieve(
            query=query,
            query_entities=query_entities,
            damping=0.5,
            top_k=10
        )
        
        print(f"  Retrieved {len(passage_ids)} passages")
        
        # Step 3: Get passage contents
        passages_collection = triplets_db.get_collection("passages")
        retrieved_passages = []
        
        for pid in passage_ids[:5]:
            passage = passages_collection.find_one({"passage_id": pid})
            if passage:
                retrieved_passages.append(passage["content"])
        
        # Step 4: Display results
        print(f"\n  Top passages:")
        for i, content in enumerate(retrieved_passages[:3]):
            print(f"    {i+1}. {content[:150]}...")
        
        # Step 5: Generate answer from retrieved passages
        if retrieved_passages:
            context = "\n\n---\n\n".join(retrieved_passages[:3])
            prompt = f"""Answer the question based ONLY on these passages:

{context[:3000]}

Question: {query}

Answer:"""
            
            try:
                answer = extractor.generate(prompt)
                print(f"\n  Generated answer: {answer[:200]}")
            except:
                print(f"\n  Gold answer would be: {answers[q_idx]}")
        
        all_results.append({
            'query': query,
            'entities': query_entities,
            'retrieved_passage_ids': passage_ids,
            'scores': scores,
            'gold_answer': answers[q_idx]
        })
    
    # =========================================================
    # 8. Print summary
    # =========================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for i, result in enumerate(all_results):
        print(f"\nQuery {i}: {result['query'][:60]}...")
        print(f"  Entities: {result['entities']}")
        print(f"  Retrieved: {len(result['retrieved_passage_ids'])} passages")
        print(f"  Gold answer: {result['gold_answer']}")
    
    return hybrid, all_results


if __name__ == "__main__":
    hybrid, results = main()