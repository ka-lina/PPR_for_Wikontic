from src.data.hotpotqa_loader import HotPotQALoader

# load hotpotQA data
loader = HotPotQALoader("./Wikontic/datasets/hotpotqa200.json")
documents, queries, answers, gold_passage_ids, question_contexts = loader.load()

print(f"Documents: {len(documents)}")
print(f"Queries: {len(queries)}")
print(f"Gold passages per query: {[len(g) for g in gold_passages[:5]]}")

llm_model_name = 'Transformers/Qwen/Qwen2.5-3B-Instruct'
        
# Create config with local models
global_config = BaseConfig(
    openie_mode='Transformers-offline',
    information_extraction_model_name=llm_model_name 
)

# Create HippoRAG instance
hipporag = HippoRAG(global_config,
    save_dir='./',
    llm_model_name=llm_model_name,
    embedding_model_name='facebook/contriever'
)

# Index documents (this creates embeddings, graph, etc.)
hipporag.index(documents)

print(hipporag.rag_qa(queries=queries,
                    gold_docs=gold_docs,
                    gold_answers=answers)[-2:])

#### TODO: metrics for HippoRAG ####