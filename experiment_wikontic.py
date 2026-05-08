from src.data.hotpotqa_loader import HotPotQALoader
from src.data.embedding_creator import EmbeddingCreator
from src.wikontic_ppr.wikontic_ppr_inference import WikonticPPRInference
from src.wikontic_ppr.graph_builder import PPRGraphBuilder
from src.wikontic_ppr.ppr_retriever import PPRRetriever
from src.evaluation import RetrievalMetrics, calculate_answer_metrics

from wikontic.utils.openai_utils import LLMTripletExtractor
from wikontic.utils.dynamic_aligner import Aligner
from wikontic.create_triplets_db import create_triplets_database

from pymongo import MongoClient

from typing import List

def save_results(retrieved_passages, answers, retrieval_metrics, 
                 gold_passages, gold_answers):
    """Save all results to JSON"""
    import json
    from datetime import datetime
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'retrieval_metrics': retrieval_metrics,
        'per_query': []
    }
    
    for i, (retrieved, answer, gold_ret, gold_ans) in enumerate(
        zip(retrieved_passages, answers, gold_passages, gold_answers)
    ):
        results['per_query'].append({
            'query_id': i,
            'retrieved_passages': retrieved[:10],  # Top 10 only
            'gold_passages': gold_ret,
            'predicted_answer': answer,
            'gold_answer': gold_ans,
            'recall@5': RetrievalMetrics.recall_at_k(retrieved, gold_ret, 5)
        })
    
    with open('wikontic_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to wikontic_experiment_results.json")

def check_existing_graph(cache_dir: str) -> bool:
    """Check if graph already exists"""
    graph_path = Path(cache_dir) / "wikontic_ppr_graph.pickle"
    metadata_path = Path(cache_dir) / "wikontic_ppr_metadata.pickle"
    return graph_path.exists() and metadata_path.exists()


def get_passage_content(passage_store, passage_ids: List[str]) -> List[str]:
    """Get actual passage text from IDs"""
    passages = []
    for pid in passage_ids:
        passage = passage_store.collection.find_one({"passage_id": pid})
        if passage:
            passages.append(passage["content"])
    return passages


if __name__ == '__main__':
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

    # Create collections for non-ontology mode
    # create_triplets_database(
    #     mongo_uri="mongodb://localhost:27018/?directConnection=true",
    #     db_name="hotpotqa_triplets",
    #     drop_collections=True
    # )

    # 3. Setup MongoDB (using Wikontic's structure)
    client = MongoClient("mongodb://localhost:27018/?directConnection=true")
    triplets_db = client["hotpotqa_triplets"]


    # 4. Initialize Wikontic components
    extractor = LLMTripletExtractor(model_name="Qwen/Qwen2.5-3B-Instruct")
    aligner = Aligner(triplets_db=triplets_db)
    embedding_model = EmbeddingCreator("facebook/contriever")

    # 5. Create PPR-enhanced Wikontic
    wikontic_ppr = WikonticPPRInference(
        extractor=extractor,
        aligner=aligner,
        triplets_db=triplets_db,
        embedding_model=embedding_model,
        cache_dir=cache_dir  
    )

    processed_ids = wikontic_ppr.passage_store.load_passage_ids()
    
    if processed_ids:
        print(f"Found {len(processed_ids)} previously processed passages")
    else:
        print("Processing documents...")
        all_passage_ids = []
        
        for q_idx, question_item in enumerate(question_passages):
            for p_idx, passage in enumerate(question_item):
                passage_id = wikontic_ppr.extract_triplets_and_add_to_db_with_passages(
                    text=passage,
                    source_text_id=f"q{q_idx}_p{p_idx}",
                )
                all_passage_ids.append(passage_id)
        
        wikontic_ppr.passage_store.save_passage_ids(all_passage_ids)
        print(f"Processed {len(all_passage_ids)} passages")
    
    # print("Processing documents...")
    # all_passage_ids = []
    
    # for q_idx, question_item in enumerate(question_contexts):
    #     for p_idx, passage in enumerate(question_item):
    #         passage_id = wikontic_ppr.extract_triplets_and_add_to_db_with_passages(
    #             text=passage,
    #             source_text_id=f"q{q_idx}_p{p_idx}",
    #         )
    #         all_passage_ids.append(passage_id)
    
    # wikontic_ppr.passage_store.save_passage_ids(all_passage_ids)
    # print(f"Processed {len(all_passage_ids)} passages")

    # 7. Build PPR graph
    graph_builder = PPRGraphBuilder(triplets_db, wikontic_ppr.passage_store, cache_dir=cache_dir)
    graph = graph_builder.build(force_rebuild=False)
    print(f"Graph built: {graph.vcount()} nodes, {graph.ecount()} edges")

    # 8. Create retriever
    retriever = PPRRetriever(
        graph=graph,
        passage_node_keys=graph_builder.passage_node_keys,
        name_to_idx=graph_builder.name_to_idx,
        embedding_model=embedding_model,
        passage_store=wikontic_ppr.passage_store,
        extractor=extractor  # Pass the LLM extractor
    )

    # Run retrieval for all queries (no gold needed here)
    all_retrieved_passages = []
    all_answers = []

    for query in queries:  # Just queries, no gold needed
        # Extract entities
        entities = extractor.extract_entities_from_question(query)
        
        # Retrieve passages (no gold_ids needed)
        passage_ids, scores = retriever.retrieve(
            query_entities=entities,
            query_text=query,
            top_k=10
        )
        passage_texts = get_passage_content(wikontic_ppr.passage_store, passage_ids)
        
        answer = retriever.answer(query, passage_ids, top_k=5)
        ################## FOR DEBUG #################
        print(f"Query: {query}")
        print(f"Entities: {entities}")
        print(f"Answer: {answer}")
        print(f"Passages: {passage_texts}")
        ###############################################
        
        all_retrieved_passages.append(passage_ids)
        all_answers.append(answer)

    # Metrics
    # 1) PPR retrieval for wikontic
    metrics = RetrievalMetrics.evaluate_all(
        retrieved_per_query=all_retrieved_passages,
        gold_per_query=gold_passage_ids,  # Same order as queries
        k_values=[1, 5, 10, 20]
    )
    
    # Answer metrics
    answer_metrics = calculate_answer_metrics(all_answers, answers)

    # Save detailed results
    save_results(all_retrieved_passages, all_answers, metrics, gold_passage_ids, answers)


    ### TODO: metrics for: ####
    # 2) original wikontic retireval for wikontic