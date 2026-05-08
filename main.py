# main.py - Complete pipeline

from pymongo import MongoClient

def main():
    # 1. Connect to MongoDB
    client = MongoClient("mongodb://localhost:27018/")
    
    # 2. Create canonical data store (single source of truth)
    canonical_store = CanonicalDataStore(client, "canonical_knowledge")
    
    # 3. Create unified extractor (choose ontology mode)
    use_ontology = False  # Set to True if you have Wikidata ontology
    extractor = UnifiedExtractor(client, use_ontology=use_ontology)
    
    # 4. Process documents ONCE
    documents = [
        "Steve Jobs co-founded Apple Inc. in 1976. Apple became a technology leader.",
        "The iPhone was introduced by Apple in 2007 and revolutionized smartphones.",
        "Steve Jobs also founded NeXT and Pixar before returning to Apple."
    ]
    
    for i, doc in enumerate(documents):
        result = extractor.process_document(doc, f"chunk_{i}", "experiment_1")
        print(f"Processed chunk {i}: {result}")
    
    # 5. Run experiments on the same extracted data
    experiment_runner = ExperimentRunner(canonical_store, extractor)
    
    questions = [
        "Who founded Apple?",
        "What company created the iPhone?",
        "What did Steve Jobs found?"
    ]
    
    results_df = experiment_runner.run_experiment(questions)
    
    # 6. Save results
    results_df.to_csv("retrieval_comparison.csv", index=False)
    print("\n" + "="*60)
    print("Results saved to retrieval_comparison.csv")
    print(results_df)

if __name__ == "__main__":
    main()