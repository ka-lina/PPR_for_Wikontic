# experiments/run_ppr_experiment.py

from pymongo import MongoClient
from src.data.hotpotqa_loader import HotPotQALoader
from src.data.embedding_creator import EmbeddingCreator
from src.evaluation import RetrievalMetrics, calculate_answer_metrics

import sys
sys.path.insert(0, "../Wikontic")
from wikontic.utils.openai_utils import LLMTripletExtractor
from wikontic.create_wikidata_ontology_db import create_wikidata_ontology_database
from wikontic.create_ontological_triplets_db import create_ontological_triplets_database


def setup_ontology_databases(mongo_uri: str):
    """Create ontology and triplets databases"""
    
    # Create Wikidata ontology database
    print("Creating Wikidata ontology database...")
    create_wikidata_ontology_database(
        mongo_uri=mongo_uri,
        database="wikidata_ontology_test",
        drop_collections=True
    )
    
    # Create ontological triplets database
    print("Creating ontological triplets database...")
    create_ontological_triplets_database(
        mongo_uri=mongo_uri,
        db_name="hotpotqa_onto_test",
        drop_collections=True
    )
    
    return "wikidata_ontology_test", "hotpotqa_onto_test"


def main():
    # 1. Setup MongoDB with ontology
    mongo_uri = "mongodb://localhost:27018/?directConnection=true"
    ontology_db_name, triplets_db_name = setup_ontology_databases(mongo_uri)
    
    client = MongoClient(mongo_uri)
    triplets_db = client[triplets_db_name]
    ontology_db = client[ontology_db_name]
    
    # 2. Load HotPotQA data
    cache_dir = "./cache/hotpotqa_experiment"
    # 1. Load data
    loader = HotPotQALoader("./Wikontic/datasets/hotpotqa200.json")
    documents, queries, answers, gold_passage_ids, question_contexts = loader.load()
    ################### FOR DEBUG ######################
    #### TODO: use full dataset for final results! #####
    num_questions = 1
    documents = documents[:num_questions * 10]  # Adjust as needed
    queries = queries[:num_questions]
    answers = answers[:num_questions]
    gold_passage_ids = gold_passage_ids[:num_questions]
    question_contexts = question_contexts[:num_questions]
    print(queries[0])
    print(answers[0])
    question_passages = []
    for j in range(len(question_contexts)):
        qp = []
        for i in gold_passage_ids[j]:
            qp.append(question_contexts[j][i])
        question_passages.append(qp)
    print(question_passages)

    ####################################################
    
    print(f"Loaded {len(documents)} documents, {len(queries)} queries")
    
    # 3. Initialize components
    print("Initializing components...")
    extractor = LLMTripletExtractor(model_name="Qwen/Qwen2.5-3B-Instruct")
    embedding_model = EmbeddingCreator("facebook/contriever")
    
    # Import structured aligner and inference
    from wikontic.utils.structured_aligner import Aligner as StructuredAligner
    from src.wikontic_ppr.structured_wikontic_ppr_inference import StructuredWikonticPPRInference
    
    aligner = StructuredAligner(triplets_db=triplets_db, ontology_db=ontology_db)
    
    # 4. Create PPR-enhanced structured Wikontic
    wikontic_ppr = StructuredWikonticPPRInference(
        extractor=extractor,
        aligner=aligner,
        triplets_db=triplets_db,
        ontology_db=ontology_db,
        embedding_model=embedding_model,
        cache_dir=cache_dir
    )
    
    # 5. Process documents with ontology filtering
    print("Processing documents with ontology...")
    for doc_id, doc_text in enumerate(documents):
        if doc_id % 20 == 0:
            print(f"  Processing doc {doc_id}/{len(documents)}")
        
        wikontic_ppr.extract_triplets_and_add_to_db_with_passages(
            text=doc_text,
            source_text_id=str(doc_id),
            sample_id="hotpotqa"
        )
    
    # 6. Build structured PPR graph
    print("Building structured PPR graph...")
    from src.wikontic_ppr.structured_graph_builder import StructuredPPRGraphBuilder
    
    graph_builder = StructuredPPRGraphBuilder(
        triplets_db=triplets_db,
        ontology_db=ontology_db,
        passage_store=wikontic_ppr.passage_store,
        cache_dir=cache_dir
    )
    graph = graph_builder.build(force_rebuild=True)
    
    print(f"Graph built: {graph.vcount()} nodes, {graph.ecount()} edges")
    
    # 7. Create retriever
    from src.wikontic_ppr.ppr_retriever import PPRRetriever
    
    retriever = PPRRetriever(
        graph=graph,
        passage_node_keys=graph_builder.passage_node_keys,
        name_to_idx=graph_builder.name_to_idx,
        embedding_model=embedding_model,
        passage_store=wikontic_ppr.passage_store,
        extractor=extractor
    )
    
    # 8. Run queries
    print("\nRunning queries...")
    all_retrieved_passages = []
    all_answers = []
    
    for q_idx, query in enumerate(queries):
        print(f"\nQuery {q_idx}: {query}")
        
        # Extract entities using ontology-aware method
        entities = wikontic_ppr.identify_relevant_entities_from_question_with_llm(
            query, use_entity_types=True
        )
        
        print(f"  Entities: {entities}")
        
        # Retrieve and answer
        result = retriever.retrieve_and_answer(
            query=query,
            query_entities=entities,
            top_k=10
        )
        
        all_retrieved_passages.append(result['passage_ids'])
        all_answers.append(result['answer'])
        
        print(f"  Answer: {result['answer'][:100]}...")
    
    # 9. Calculate metrics
    print("\n" + "="*50)
    print("RETRIEVAL METRICS")
    print("="*50)
    
    retrieval_metrics = RetrievalMetrics.evaluate_all(
        retrieved_per_query=all_retrieved_passages,
        gold_per_query=gold_passage_ids,
        k_values=[1, 5, 10, 20]
    )
    
    for metric, value in retrieval_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n" + "="*50)
    print("ANSWER METRICS")
    print("="*50)
    
    answer_metrics = calculate_answer_metrics(all_answers, answers)
    for metric, value in answer_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 10. Save results
    import json
    from datetime import datetime
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'ontology',
        'retrieval_metrics': retrieval_metrics,
        'answer_metrics': answer_metrics,
        'per_query': []
    }
    
    for i, (query, retrieved, answer, gold_passages, gold_answer) in enumerate(
        zip(queries, all_retrieved_passages, all_answers, gold_passage_ids, answers)
    ):
        results['per_query'].append({
            'query_id': i,
            'query': query,
            'retrieved_passages': retrieved[:10],
            'gold_passages': gold_passages,
            'predicted_answer': answer,
            'gold_answer': gold_answer
        })
    
    with open('ontology_ppr_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to ontology_ppr_results.json")


if __name__ == "__main__":
    main()