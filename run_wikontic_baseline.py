import os
import sys

from pymongo import MongoClient

sys.path.insert(0, "../Wikontic")

from hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from hipporag.evaluation.retrieval_eval import RetrievalRecall
from src.data.hotpotqa_loader import HotPotQALoader
from wikontic.utils.openai_utils import LLMTripletExtractor


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def get_question_indices(total: int) -> list[int]:
    if os.environ.get("QUESTION_INDICES"):
        return [int(i.strip()) for i in os.environ["QUESTION_INDICES"].split(",") if i.strip()]

    start = env_int("QUESTION_INDEX", 0)
    num_questions = env_int("NUM_QUESTIONS", 200)
    return list(range(start, min(start + num_questions, total)))


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


def make_wikontic_inference(extractor, triplets_db, ontology_db, use_ontology: bool):
    if use_ontology:
        from wikontic.utils.structured_aligner import Aligner as StructuredAligner
        from wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB

        aligner = StructuredAligner(ontology_db=ontology_db, triplets_db=triplets_db)
        return StructuredInferenceWithDB(extractor, aligner, triplets_db)

    from wikontic.utils.dynamic_aligner import Aligner
    from wikontic.utils.inference_with_db import InferenceWithDB

    aligner = Aligner(triplets_db=triplets_db)
    return InferenceWithDB(extractor, aligner, triplets_db)


def unique_in_order(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def get_gold_docs_for_question(q_idx: int, documents, gold_passage_ids, question_contexts) -> list[str]:
    gold_docs = unique_in_order([documents[i] for i in gold_passage_ids[q_idx]])
    if os.environ.get("CORPUS_MODE", "gold") == "question_context":
        return unique_in_order(question_contexts[q_idx])
    return gold_docs


def entities_for_passage_retrieval(linked_entities: list[str], supporting_triplets: list[dict]) -> list[str]:
    ordered: list[str] = []
    seen = set()
    for ent in linked_entities:
        if ent and ent not in seen:
            seen.add(ent)
            ordered.append(ent)
    for triplet in supporting_triplets:
        for key in ("subject", "object"):
            ent = triplet.get(key)
            if ent and ent not in seen:
                seen.add(ent)
                ordered.append(ent)
    return ordered


def retrieve_passages_from_entity_edges(
    triplets_db,
    entity_names: list[str],
    sample_id: str | None,
    limit: int,
) -> list[str]:
    """Rank passages by summed passage–entity edge weights (Wikontic graph retrieval proxy)."""
    edges_coll = triplets_db.get_collection("passage_entity_edges")
    pass_coll = triplets_db.get_collection("passages")
    passage_score: dict[str, float] = {}
    for ent in entity_names:
        if not ent:
            continue
        for edge in edges_coll.find({"entity_name": ent}):
            pid = edge.get("passage_id")
            if not pid:
                continue
            doc = pass_coll.find_one({"passage_id": pid}, {"source_doc_id": 1})
            if not doc:
                continue
            if sample_id and doc.get("source_doc_id") != sample_id:
                continue
            w = float(edge.get("weight", 1.0))
            passage_score[pid] = passage_score.get(pid, 0.0) + w
    ranked_ids = sorted(passage_score.keys(), key=lambda p: passage_score[p], reverse=True)
    contents: list[str] = []
    seen_content: set[str] = set()
    for pid in ranked_ids:
        doc = pass_coll.find_one({"passage_id": pid})
        if not doc:
            continue
        text = doc.get("content")
        if text and text not in seen_content:
            seen_content.add(text)
            contents.append(text)
        if len(contents) >= limit:
            break
    if len(contents) < limit:
        filt = {"source_doc_id": sample_id} if sample_id else {}
        for doc in pass_coll.find(filt).sort([("source_doc_id", 1), ("chunk_index", 1)]):
            text = doc.get("content")
            if text and text not in seen_content:
                seen_content.add(text)
                contents.append(text)
            if len(contents) >= limit:
                break
    return contents


def clean_answer(answer) -> str:
    if answer is None:
        return ""
    answer = str(answer).strip()
    if answer.lower() in {"none", "null", "n/a"}:
        return ""
    if "Answer:" in answer:
        answer = answer.rsplit("Answer:", 1)[-1].strip()
    return answer.strip().strip('"').strip("'").strip()


def main():
    use_ontology = env_bool("USE_ONTOLOGY", False)
    sample_scoped = env_bool("WIKONTIC_SAMPLE_SCOPED", True)
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27018/?directConnection=true")
    triplets_db_name = os.environ.get("TRIPLETS_DB", "hotpotqa_hybrid_ontology" if use_ontology else "hotpotqa_hybrid")
    ontology_db_name = os.environ.get("WIKIDATA_ONTOLOGY_DB", "wikidata_ontology_dev")

    client = MongoClient(mongo_uri)
    triplets_db = client[triplets_db_name]
    ontology_db = client[ontology_db_name] if use_ontology else None

    if triplets_db.get_collection("triplets").count_documents({}) == 0:
        raise RuntimeError(f"Triplets DB '{triplets_db_name}' is empty. Run extraction first.")

    loader = HotPotQALoader(os.environ.get("HOTPOTQA_PATH", "../Wikontic/datasets/hotpotqa200.json"))
    documents, queries, answers, gold_passage_ids, question_contexts = loader.load()
    question_indices = get_question_indices(len(queries))

    recall_k_list = [int(x) for x in os.environ.get("RECALL_K_LIST", "1,5,10,20").split(",") if x.strip()]
    if not recall_k_list:
        recall_k_list = [1, 5, 10, 20]
    recall_top = max(recall_k_list)

    print("Mode: pure Wikontic")
    print(f"Ontology: {use_ontology}")
    print(f"Triplets DB: {triplets_db_name}")
    print(f"Sample scoped: {sample_scoped}")
    multi_step_qa = env_bool("WIKONTIC_MULTI_STEP_QA", False)
    print(f"multi_step_qa (answer_with_qa_collapsing): {multi_step_qa}")
    print(f"use_qualifiers: {env_bool('WIKONTIC_USE_QUALIFIERS', False)}")
    print(f"use_filtered_triplets (ontology_filtered): {env_bool('WIKONTIC_USE_FILTERED_TRIPLETS', False)}")
    if env_bool("WIKONTIC_USE_FILTERED_TRIPLETS", False) and not use_ontology:
        print(
            "WARNING: WIKONTIC_USE_FILTERED_TRIPLETS only applies with USE_ONTOLOGY=1 "
            "(dynamic Aligner has no ontology_filtered_triplets collection)."
        )
    print(f"Question indices: {question_indices}")

    extractor = make_extractor()
    wikontic = make_wikontic_inference(extractor, triplets_db, ontology_db, use_ontology)

    predicted_answers = []
    gold_answers = []
    gold_docs_by_query: list[list[str]] = []
    retrieved_docs_by_query: list[list[str]] = []
    supporting_triplet_counts = []

    for q_idx in question_indices:
        query = queries[q_idx]
        sample_id = f"q{q_idx}" if sample_scoped else None

        print(f"\nQuery {q_idx}: {query}")
        print(f"Gold answer: {answers[q_idx]}")

        linked_entities = wikontic.identify_relevant_entities_from_question_with_llm(
            query,
            sample_id=sample_id,
            use_entity_types=use_ontology,
        )
        if multi_step_qa:
            answer = wikontic.answer_with_qa_collapsing(
                query,
                sample_id=sample_id,
                max_attempts=env_int("WIKONTIC_QA_COLLAPSE_MAX_ATTEMPTS", 5),
                use_qualifiers=env_bool("WIKONTIC_USE_QUALIFIERS", False),
                use_filtered_triplets=env_bool("WIKONTIC_USE_FILTERED_TRIPLETS", False),
            )
            supporting_triplets = []
        else:
            supporting_triplets, answer = wikontic.answer_question_with_llm(
                question=query,
                linked_entities=linked_entities,
                sample_id=sample_id,
                hop_depth=env_int("WIKONTIC_HOP_DEPTH", 5),
                use_filtered_triplets=env_bool("WIKONTIC_USE_FILTERED_TRIPLETS", False),
                use_qualifiers=env_bool("WIKONTIC_USE_QUALIFIERS", False),
            )

        answer = clean_answer(answer)
        predicted_answers.append(answer)
        gold_answers.append([answers[q_idx]])
        gold_docs_by_query.append(
            get_gold_docs_for_question(q_idx, documents, gold_passage_ids, question_contexts)
        )
        ent_for_ret = entities_for_passage_retrieval(linked_entities, supporting_triplets)
        retrieved_docs_by_query.append(
            retrieve_passages_from_entity_edges(triplets_db, ent_for_ret, sample_id, limit=recall_top)
        )
        supporting_triplet_counts.append(len(supporting_triplets))

        print(f"Linked entities: {linked_entities}")
        if multi_step_qa:
            print("Supporting triplets: n/a (multi-step QA uses internal 1-hop rounds)")
        else:
            print(f"Supporting triplets: {len(supporting_triplets)}")
        print(f"Predicted answer: {answer}")

    retrieval_metrics, _ = RetrievalRecall(global_config=None).calculate_metric_scores(
        gold_docs=gold_docs_by_query,
        retrieved_docs=retrieved_docs_by_query,
        k_list=recall_k_list,
    )
    print("\nPure Wikontic retrieval metrics (passages ranked by passage_entity_edges × linked/subgraph entities):")
    for name, value in retrieval_metrics.items():
        print(f"  {name}: {value}")

    exact_match, _ = QAExactMatch().calculate_metric_scores(
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
    )
    f1, _ = QAF1Score().calculate_metric_scores(
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
    )

    print("\nPure Wikontic QA metrics:")
    for name, value in {**exact_match, **f1}.items():
        print(f"  {name}: {value}")
    if multi_step_qa:
        print("  AvgSupportingTriplets: n/a (multi-step QA; see Wikontic qa_eval_hotpot multi_step_qa path)")
    elif supporting_triplet_counts:
        avg_triplets = sum(supporting_triplet_counts) / len(supporting_triplet_counts)
        print(f"  AvgSupportingTriplets: {avg_triplets}")


if __name__ == "__main__":
    main()
