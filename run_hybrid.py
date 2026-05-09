# run_hybrid.py - simplified

from src.data.hotpotqa_loader import HotPotQALoader
from src.hybrid.passage_manager import PassageManager
from src.hybrid.wikontic_hipporag import WikonticHippoRAG
from src.data.embedding_creator import EmbeddingCreator
from src.wikontic_ppr.wikontic_ppr_inference import WikonticPPRInference

from pymongo import MongoClient
import sys
sys.path.insert(0, "../Wikontic")
from wikontic.utils.dynamic_aligner import Aligner
from wikontic.utils.openai_utils import LLMTripletExtractor
from hipporag.utils.config_utils import BaseConfig


def main():
    # 1. Setup MongoDB
    client = MongoClient("mongodb://localhost:27018/?directConnection=true")
    client.drop_database("hotpotqa_hybrid")
    triplets_db = client["hotpotqa_hybrid"]
    
    # 2. Load data
    loader = HotPotQALoader("./Wikontic/datasets/hotpotqa200.json")
    documents, queries, answers, gold_passage_ids, question_contexts = loader.load()
    
    # Use first question
    q_idx = 5
    query = queries[q_idx]
    
    # Get relevant passages for this question
    passages = []
    for i in gold_passage_ids[q_idx]:
        passages.append(documents[i])
        # if i < len(question_contexts[q_idx]):
        #     passages.append(question_contexts[q_idx][i])
    
    print(f"Query: {query}")
    print(f"Gold answer: {answers[q_idx]}")
    print(f"Passages: {len(passages)}")
    print(gold_passage_ids[q_idx])
    
    # 3. Initialize models
    embedding_model = EmbeddingCreator("facebook/contriever")
    extractor = LLMTripletExtractor(model_name="Qwen/Qwen2.5-3B-Instruct")
    aligner = Aligner(triplets_db=triplets_db)
    
    # 4. Create passage database and extract triplets
    passage_manager = PassageManager(triplets_db, embedding_model)
    edges_collection = triplets_db.get_collection("passage_entity_edges")
    
    wikontic = WikonticPPRInference(extractor, aligner, triplets_db, embedding_model)
    
    for p_idx, passage in enumerate(passages):
        print(f"\nProcessing passage {p_idx}...")
        
        # Add passage
        passage_id = passage_manager.add_passage(passage, f"q{q_idx}", p_idx)
        
        # Extract triplets
        initial, final, filtered = wikontic.extract_triplets(
            passage, sample_id=f"q{q_idx}", source_text_id=f"p{p_idx}"
        )
        
        if final:
            aligner.add_triplets(final, sample_id=f"q{q_idx}")
            
            for triplet in final:
                if triplet.get("subject"):
                    edges_collection.update_one(
                        {"passage_id": passage_id, "entity_name": triplet["subject"]},
                        {"$set": {"weight": 1.0}}, upsert=True
                    )
                if triplet.get("object"):
                    edges_collection.update_one(
                        {"passage_id": passage_id, "entity_name": triplet["object"]},
                        {"$set": {"weight": 1.0}}, upsert=True
                    )
            
            print(f"  Added {len(final)} triplets")
    
    # 5. Build WikonticHippoRAG index
    llm_model_name = 'Transformers/Qwen/Qwen2.5-3B-Instruct'
        
    # Create config with local models
    global_config = BaseConfig(
        openie_mode='Transformers-offline',
        information_extraction_model_name=llm_model_name 
    )
    hybrid = WikonticHippoRAG(triplets_db, embedding_model,
        global_config,
        save_dir='./',
        llm_model_name=llm_model_name,
        embedding_model_name='facebook/contriever')
    
    # Get passage texts
    passage_texts = [passage_manager.get_passage(p["passage_id"])["content"] 
                     for p in passage_manager.passages.find()]
    
    hybrid.index_from_wikontic(passage_texts)
    
    # 6. Retrieve
    results = hybrid.retrieve([query], num_to_retrieve=5)
    # results = hybrid.retrieve_from_wikontic([query], num_to_retrieve=5)
    
    for result in results:
        print(f"\nQuery: {result.question}")
        print(f"Retrieved {len(result.docs)} passages")
        for i, (doc, score) in enumerate(zip(result.docs[:3], result.doc_scores[:3])):
            print(f"  {i+1}. [{score:.4f}] {doc[:100]}...")
    
    (queries_solutions, all_response_message, all_metadata, \
    overall_retrieval_result, overall_qa_results) = hybrid.rag_qa(queries=[query],
                    gold_docs=[passages],
                    gold_answers=[answers[q_idx]])
    print('queries_solutions answers', [qa_result.answer for qa_result in queries_solutions])
    # print('all_response_message', all_response_message)
    print('overall_retrieval_result', overall_retrieval_result)
    print('overall_qa_results', overall_qa_results)


if __name__ == "__main__":
    main()