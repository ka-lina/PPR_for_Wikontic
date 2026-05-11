import os
import sys

from pymongo import MongoClient

from src.data.embedding_creator import EmbeddingCreator
from src.data.hotpotqa_loader import HotPotQALoader
from src.hybrid.passage_manager import PassageManager
from src.hybrid.wikontic_hipporag import WikonticHippoRAG, normalize_graph_value
from src.wikontic_ppr.wikontic_ppr_inference import WikonticPPRInference

sys.path.insert(0, "../Wikontic")

from hipporag.utils.config_utils import BaseConfig
from hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from hipporag.evaluation.retrieval_eval import RetrievalRecall
from wikontic.utils.openai_utils import LLMTripletExtractor


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def make_extractor() -> LLMTripletExtractor:
    llm_base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key = os.environ.get("OPENAI_API_KEY")
    if not llm_api_key:
        if "localhost" in llm_base_url or "127.0.0.1" in llm_base_url:
            llm_api_key = "local"
        else:
            raise RuntimeError("Set OPENAI_API_KEY or use a local OpenAI-compatible LLM_BASE_URL")

    return LLMTripletExtractor(
        api_key=llm_api_key,
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        base_url=llm_base_url,
    )


def get_question_indices(total: int) -> list[int]:
    if os.environ.get("QUESTION_INDICES"):
        return [int(i.strip()) for i in os.environ["QUESTION_INDICES"].split(",") if i.strip()]

    start = env_int("QUESTION_INDEX", 5)
    num_questions = env_int("NUM_QUESTIONS", 1)
    return list(range(start, min(start + num_questions, total)))


def unique_in_order(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def get_passages_for_question(q_idx: int, documents, gold_passage_ids, question_contexts) -> tuple[list[str], list[str]]:
    gold_docs = unique_in_order([documents[i] for i in gold_passage_ids[q_idx]])
    if os.environ.get("CORPUS_MODE", "gold") == "question_context":
        passages = unique_in_order(question_contexts[q_idx])
    else:
        passages = gold_docs
    return passages, gold_docs


def setup_databases(client: MongoClient, mongo_uri: str, use_ontology: bool, reuse_db: bool):
    triplets_db_name = os.environ.get("TRIPLETS_DB", "hotpotqa_hybrid")
    if not reuse_db:
        client.drop_database(triplets_db_name)

    if not use_ontology:
        return client[triplets_db_name], None

    ontology_db_name = os.environ.get("WIKIDATA_ONTOLOGY_DB", "wikidata_ontology_dev")
    ontology_db = client[ontology_db_name]

    if ontology_db.get_collection("entity_types").estimated_document_count() == 0:
        if not env_bool("CREATE_ONTOLOGY_DB", False):
            raise RuntimeError(
                f"Ontology DB '{ontology_db_name}' is empty. Set CREATE_ONTOLOGY_DB=1 to populate it."
            )
        from wikontic.create_wikidata_ontology_db import create_wikidata_ontology_database

        create_wikidata_ontology_database(
            mongo_uri=mongo_uri,
            database=ontology_db_name,
            drop_collections=False,
        )

    from wikontic.create_ontological_triplets_db import create_ontological_triplets_database

    if reuse_db:
        triplets_db = client[triplets_db_name]
    else:
        triplets_db = create_ontological_triplets_database(
            mongo_uri=mongo_uri,
            db_name=triplets_db_name,
            drop_collections=True,
        )
    return triplets_db, ontology_db


def make_wikontic_components(extractor, triplets_db, ontology_db, embedding_model, use_ontology: bool):
    if use_ontology:
        from wikontic.utils.structured_aligner import Aligner as StructuredAligner
        from src.wikontic_ppr.structured_wikontic_ppr_inference import StructuredWikonticPPRInference

        aligner = StructuredAligner(ontology_db=ontology_db, triplets_db=triplets_db)
        wikontic = StructuredWikonticPPRInference(
            extractor=extractor,
            aligner=aligner,
            triplets_db=triplets_db,
            ontology_db=ontology_db,
            embedding_model=embedding_model,
        )
        return aligner, wikontic

    from wikontic.utils.dynamic_aligner import Aligner

    aligner = Aligner(triplets_db=triplets_db)
    wikontic = WikonticPPRInference(extractor, aligner, triplets_db, embedding_model)
    return aligner, wikontic


def record_collection_counts(triplets_db):
    names = [
        "passages",
        "passage_entity_edges",
        "initial_triplets",
        "triplets",
        "filtered_triplets",
        "ontology_filtered_triplets",
    ]
    print("\nCollection counts:")
    for name in names:
        if name in triplets_db.list_collection_names():
            print(f"  {name}: {triplets_db.get_collection(name).count_documents({})}")


def add_passage_edges(edges_collection, passage_id: str, triplets: list[dict]):
    for triplet in triplets:
        for role, type_key in [("subject", "subject_type"), ("object", "object_type")]:
            entity = normalize_graph_value(triplet.get(role)).strip()
            if not entity:
                continue
            edges_collection.update_one(
                {"passage_id": passage_id, "entity_name": entity},
                {"$set": {"weight": 1.0, "entity_type": normalize_graph_value(triplet.get(type_key)).strip()}},
                upsert=True,
            )


def process_passage(wikontic, aligner, passage_manager, edges_collection, passage: str, sample_id: str, p_idx: int, use_ontology: bool):
    passage_id = passage_manager.add_passage(passage, sample_id, p_idx)
    source_text_id = f"{sample_id}_p{p_idx}"

    if use_ontology:
        initial, final, filtered, ontology_filtered = wikontic.extract_triplets_with_ontology_filtering(
            passage, sample_id=sample_id, source_text_id=source_text_id
        )
        if initial:
            aligner.add_initial_triplets(initial, sample_id=sample_id)
        if filtered:
            aligner.add_filtered_triplets(filtered, sample_id=sample_id)
        if ontology_filtered:
            aligner.add_ontology_filtered_triplets(ontology_filtered, sample_id=sample_id)
    else:
        initial, final, filtered = wikontic.extract_triplets(
            passage, sample_id=sample_id, source_text_id=source_text_id
        )
        if initial:
            aligner.add_initial_triplets(initial, sample_id=sample_id)
        if filtered:
            aligner.add_filtered_triplets(filtered, sample_id=sample_id)

    if not final and initial and env_bool("USE_INITIAL_TRIPLETS", True):
        print(f"  Using {len(initial)} initial triplets because refinement returned none")
        final = initial

    if final:
        aligner.add_triplets(final, sample_id=sample_id)
        add_passage_edges(edges_collection, passage_id, final)
        print(f"  Added {len(final)} triplets")

    return passage_id, final


def build_hybrid(triplets_db, embedding_model):
    llm_model_name = os.environ.get("HIPPO_LLM_MODEL", "Transformers/Qwen/Qwen2.5-3B-Instruct")
    global_config = BaseConfig(
        openie_mode="Transformers-offline",
        information_extraction_model_name=llm_model_name,
        retrieval_top_k=env_int("RETRIEVAL_TOP_K", 5),
        linking_top_k=env_int("LINKING_TOP_K", 5),
        damping=float(os.environ.get("DAMPING", "0.5")),
    )
    global_config.skip_openie = env_bool("SKIP_QA", True)

    return WikonticHippoRAG(
        triplets_db,
        embedding_model,
        global_config,
        save_dir=os.environ.get("HIPPO_SAVE_DIR", "./"),
        llm_model_name=llm_model_name,
        embedding_model_name=os.environ.get("EMBEDDING_MODEL", "facebook/contriever"),
        external_embedding_model=embedding_model,
    )


def validate_reusable_db(triplets_db):
    missing = [
        name
        for name in ("passages", "triplets")
        if triplets_db.get_collection(name).count_documents({}) == 0
    ]
    if missing:
        raise RuntimeError(
            "REUSE_DB=1 requires an existing populated database. "
            f"Empty collections: {', '.join(missing)}"
        )


def main():
    use_ontology = env_bool("USE_ONTOLOGY", False)
    reuse_db = env_bool("REUSE_DB", False)
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27018/?directConnection=true")
    client = MongoClient(mongo_uri)
    triplets_db, ontology_db = setup_databases(client, mongo_uri, use_ontology, reuse_db)

    data_path = os.environ.get("HOTPOTQA_PATH", "../Wikontic/datasets/hotpotqa200.json")
    loader = HotPotQALoader(data_path)
    documents, queries, answers, gold_passage_ids, question_contexts = loader.load()

    question_indices = get_question_indices(len(queries))
    print(f"Mode: {'ontology' if use_ontology else 'non-ontology'}")
    print(f"Reuse DB: {reuse_db}")
    print(f"Question indices: {question_indices}")

    embedding_device = os.environ.get("EMBEDDING_DEVICE")
    if embedding_device is None:
        embedding_device = "cpu" if not env_bool("SKIP_QA", True) else "cuda"
    embedding_model = EmbeddingCreator(
        os.environ.get("EMBEDDING_MODEL", "facebook/contriever"),
        device=embedding_device,
    )
    extractor = None
    aligner = None
    wikontic = None
    if reuse_db:
        validate_reusable_db(triplets_db)
        print("Reusing existing passages and triplets; skipping extraction.")
    else:
        extractor = make_extractor()
        aligner, wikontic = make_wikontic_components(extractor, triplets_db, ontology_db, embedding_model, use_ontology)

    passage_manager = PassageManager(triplets_db, embedding_model)
    edges_collection = triplets_db.get_collection("passage_entity_edges")

    gold_docs_by_query = []
    for q_idx in question_indices:
        query = queries[q_idx]
        passages, gold_docs = get_passages_for_question(q_idx, documents, gold_passage_ids, question_contexts)
        gold_docs_by_query.append(gold_docs)

        print(f"\nQuery {q_idx}: {query}")
        print(f"Gold answer: {answers[q_idx]}")
        print(f"Passages: {len(passages)}")
        print(gold_passage_ids[q_idx])

        if reuse_db:
            continue

        for p_idx, passage in enumerate(passages):
            print(f"\nProcessing passage {p_idx}...")
            process_passage(
                wikontic,
                aligner,
                passage_manager,
                edges_collection,
                passage,
                sample_id=f"q{q_idx}",
                p_idx=p_idx,
                use_ontology=use_ontology,
            )

    record_collection_counts(triplets_db)

    hybrid = build_hybrid(triplets_db, embedding_model)
    passage_texts = [
        p["content"]
        for p in passage_manager.passages.find().sort([("source_doc_id", 1), ("chunk_index", 1)])
    ]
    hybrid.index_from_wikontic(passage_texts)

    eval_queries = [queries[i] for i in question_indices]
    results = hybrid.retrieve(eval_queries, num_to_retrieve=env_int("NUM_TO_RETRIEVE", 5))

    for result in results:
        print(f"\nQuery: {result.question}")
        print(f"Retrieved {len(result.docs)} passages")
        for i, (doc, score) in enumerate(zip(result.docs[:3], result.doc_scores[:3])):
            print(f"  {i+1}. [{score:.4f}] {doc[:100]}...")

    retrieval_metrics, _ = RetrievalRecall(global_config=hybrid.global_config).calculate_metric_scores(
        gold_docs=gold_docs_by_query,
        retrieved_docs=[result.docs for result in results],
        k_list=[1, 5, 10, 20],
    )
    print("\nRetrieval metrics:")
    for name, value in retrieval_metrics.items():
        print(f"  {name}: {value}")

    if env_bool("SKIP_QA", True):
        print("Skipping rag_qa because SKIP_QA=1")
        return

    gold_answers = [[answers[i]] for i in question_indices]
    (queries_solutions, all_response_message, all_metadata, \
    overall_retrieval_result, overall_qa_results) = hybrid.rag_qa(
        queries=results,
        gold_docs=gold_docs_by_query,
        gold_answers=gold_answers,
    )
    predicted_answers = [qa_result.answer for qa_result in queries_solutions]
    qa_exact_match, _ = QAExactMatch(global_config=hybrid.global_config).calculate_metric_scores(
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
    )
    qa_f1, _ = QAF1Score(global_config=hybrid.global_config).calculate_metric_scores(
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
    )

    print('queries_solutions answers sample', predicted_answers[:5])
    print('overall_retrieval_result', overall_retrieval_result)
    print('overall_qa_results', overall_qa_results)
    print("\nQA metrics:")
    for name, value in {**qa_exact_match, **qa_f1}.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()