import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
# HippoRAG: sibling repo …/HippoRAG/src (same layout as PPR_for_Wikontic)
_study_root = _repo_root.parent
_hippo_src = _study_root / "HippoRAG" / "src"
if _hippo_src.is_dir() and str(_hippo_src) not in sys.path:
    sys.path.insert(0, str(_hippo_src))

from tqdm import tqdm
import argparse
import warnings
import re
from unidecode import unidecode
from pymongo.mongo_client import MongoClient
import string
import os

from wikontic.utils.structured_aligner import Aligner as StructuredDBAligner
from wikontic.utils.dynamic_aligner import Aligner as DynamicDBAligner
from wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
from wikontic.utils.inference_with_db import InferenceWithDB
from wikontic.utils.openai_utils import LLMTripletExtractor

import logging
import jsonlines
from dotenv import load_dotenv, find_dotenv

logger = logging.getLogger("QAEvalHotpot")
logger.setLevel(logging.ERROR)

warnings.filterwarnings("ignore")
_ = load_dotenv(find_dotenv())


def normalize(input_string):
    input_string = unidecode(input_string)
    input_string = input_string.lower()

    # Remove commas and periods between digits (e.g., 7,531 or 7.531 -> 7531)
    input_string = re.sub(r"(?<=\d)[,\.](?=\d)", "", input_string)

    # Replace all other punctuation with a space
    input_string = re.sub(f"[{re.escape(string.punctuation)}]", " ", input_string)

    # Replace multiple spaces with a single space
    input_string = re.sub(r"\s+", " ", input_string)

    # Trim leading/trailing whitespace
    return input_string.strip()


def _qa_metrics_local(gold_lists: list, preds: list) -> dict:
    from collections import Counter

    def _f1_pair(g: str, p: str) -> float:
        gt, pt = normalize(g).split(), normalize(p).split()
        if not pt or not gt:
            return 0.0
        c = Counter(pt) & Counter(gt)
        n = sum(c.values())
        if n == 0:
            return 0.0
        prec, rec = n / len(pt), n / len(gt)
        return 2 * prec * rec / (prec + rec)

    em = f1 = 0.0
    n = len(gold_lists)
    if n == 0:
        return {"ExactMatch": 0.0, "F1": 0.0}
    for golds, pred in zip(gold_lists, preds):
        best_em = max(
            1.0 if normalize(pred) == normalize(g) else 0.0 for g in golds
        )
        best_f1 = max(_f1_pair(g, pred) for g in golds)
        em += best_em
        f1 += best_f1
    return {"ExactMatch": em / n, "F1": f1 / n}


def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    return client


def get_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        ds = json.load(f)
    return ds


def _unique_in_order(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _gold_passage_texts_hotpot_item(item: dict, corpus_mode: str) -> list[str]:
    """Gold (supporting) passage texts or full question context, aligned with run_wikontic_baseline."""
    supporting_titles = {sf[0] for sf in item.get("supporting_facts", []) if sf}
    if corpus_mode == "question_context":
        return _unique_in_order(
            [" ".join(sents) for _, sents in item.get("context", []) if sents]
        )
    gold_texts = []
    seen = set()
    for ctx_title, ctx_sentences in item.get("context", []):
        if ctx_title in supporting_titles:
            text = " ".join(ctx_sentences)
            if text and text not in seen:
                seen.add(text)
                gold_texts.append(text)
    return gold_texts


def _entities_for_passage_retrieval(
    linked_entities: list, supporting_triplets: list[dict]
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for ent in linked_entities or []:
        if ent and ent not in seen:
            seen.add(ent)
            ordered.append(str(ent))
    for triplet in supporting_triplets or []:
        if not isinstance(triplet, dict):
            continue
        for key in ("subject", "object"):
            ent = triplet.get(key)
            if ent is None:
                continue
            if isinstance(ent, (list, tuple, set)):
                for x in ent:
                    s = str(x).strip()
                    if s and s not in seen:
                        seen.add(s)
                        ordered.append(s)
            else:
                s = str(ent).strip()
                if s and s not in seen:
                    seen.add(s)
                    ordered.append(s)
    return ordered


def _retrieve_passages_from_entity_edges(
    triplets_db,
    entity_names: list[str],
    sample_id: str | None,
    limit: int,
) -> list[str]:
    if triplets_db.get_collection("passages").estimated_document_count() == 0:
        return []
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
    ranked_ids = sorted(
        passage_score.keys(), key=lambda p: passage_score[p], reverse=True
    )
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


def _retrieve_passages_hotpot_context_overlap(
    item: dict,
    entity_names: list[str],
    limit: int,
) -> list[str]:
    """
    When Mongo has no `passages` collection: rank Hotpot context paragraphs by
    substring overlap with Wikontic-linked entity strings (retrieval proxy).
    """
    passages: list[str] = []
    for _, ctx_sentences in item.get("context", []):
        if ctx_sentences:
            passages.append(" ".join(ctx_sentences))
    if not passages:
        return []
    if not entity_names:
        return passages[:limit]
    ents = [e.strip().lower() for e in entity_names if e and str(e).strip()]
    scored: list[tuple[float, str]] = []
    for text in passages:
        tl = text.lower()
        score = sum(1 for e in ents if len(e) >= 2 and e in tl)
        scored.append((float(score), text))
    scored.sort(key=lambda x: x[0], reverse=True)
    ordered = [t for _, t in scored]
    return ordered[:limit]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mongo_uri",
        type=str,
        default=os.environ.get(
            "MONGO_URI",
            "mongodb://localhost:27018/?directConnection=true",
        ),
    )
    parser.add_argument("--ontology_db_name", type=str, default="wikidata_ontology")
    parser.add_argument(
        "--triplets_db_name",
        type=str,
        default=None,
        help="Mongo triplets DB (default: triplets_db_<LLM_MODEL>_<onto|non_onto>, same as dataset_inference.py)",
    )
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Overridden by env LLM_MODEL if set")
    parser.add_argument("--dataset_path", type=str, default="datasets/hotpotqa200.json")
    parser.add_argument(
        "--structured_inference",
        action="store_true",
        help="Enable structured inference",
    )
    parser.add_argument(
        "--no_structured_inference",
        action="store_false",
        dest="structured_inference",
        help="Disable structured inference",
    )
    parser.add_argument(
        "--multi-step-qa", action="store_true", help="Enable multi-step QA"
    )
    parser.add_argument(
        "--no_multi-step-qa",
        action="store_false",
        dest="multi_step_qa",
        help="Disable multi-step QA",
    )
    parser.add_argument("--use_qualifiers", action="store_true", help="Use qualifiers")
    parser.add_argument(
        "--no_use_qualifiers",
        action="store_false",
        dest="use_qualifiers",
        help="Disable use of qualifiers",
    )
    parser.add_argument(
        "--use_filtered_triplets", action="store_true", help="Use filtered triplets"
    )
    parser.add_argument(
        "--no_use_filtered_triplets",
        action="store_false",
        dest="use_filtered_triplets",
        help="Disable use of filtered triplets",
    )
    parser.add_argument("--run_number", type=int, default=1, help="Run number")
    parser.set_defaults(structured_inference=False)
    parser.set_defaults(multi_step_qa=False)
    parser.set_defaults(use_qualifiers=True)
    parser.set_defaults(use_filtered_triplets=False)
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    model_name = os.getenv("LLM_MODEL") or args.model_name
    model_tag = model_name.replace("/", "_").replace(".", "_").replace(":", "_")
    suffix = "_onto" if args.structured_inference else "_non_onto"
    triplets_db_name = args.triplets_db_name or f"triplets_db_{model_tag}{suffix}"

    mongo_client = get_mongo_client(args.mongo_uri)
    db = mongo_client.get_database(args.ontology_db_name)

    triplets_db = mongo_client.get_database(triplets_db_name)
    ontology_db = mongo_client.get_database(args.ontology_db_name)

    api_key = (
        os.getenv("OPENROUTER_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    proxy_url = os.getenv("PROXY_URL")
    base_url = (
        os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    if not api_key and (
        "localhost" in base_url or "127.0.0.1" in base_url
    ):
        api_key = "local"

    dataset_path = args.dataset_path
    use_qualifiers = args.use_qualifiers
    use_filtered_triplets = args.use_filtered_triplets

    logger.info(f"Use qualifier: {args.use_qualifiers}")
    logger.info(f"Use filtered triplets: {args.use_filtered_triplets}")

    ds = get_dataset(dataset_path)

    id2sample = {}
    for elem in ds:
        id2sample[elem["_id"]] = elem

    if args.structured_inference:
        logger.info("Structured inference enabled")
        aligner = StructuredDBAligner(ontology_db=ontology_db, triplets_db=triplets_db)
    else:
        logger.info("Structured inference disabled, using dynamic inference")
        aligner = DynamicDBAligner(triplets_db=triplets_db)

    extractor = LLMTripletExtractor(
        model=model_name, api_key=api_key, proxy=proxy_url, base_url=base_url
    )

    if args.structured_inference:
        inference_with_db = StructuredInferenceWithDB(
            extractor=extractor, aligner=aligner, triplets_db=triplets_db
        )
    else:
        inference_with_db = InferenceWithDB(
            extractor=extractor, aligner=aligner, triplets_db=triplets_db
        )

    unique_sample_ids = triplets_db.get_collection("triplets").distinct("sample_id")

    if args.multi_step_qa:
        logger.info(f"Enabled multi-step QA")
    else:
        logger.info(f"Disabled multi-step QA")

    corpus_mode = os.environ.get("CORPUS_MODE", "gold")
    sample_scoped = os.environ.get("WIKONTIC_SAMPLE_SCOPED", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    recall_k_list = [
        int(x)
        for x in os.environ.get("RECALL_K_LIST", "1,5,10,20").split(",")
        if x.strip()
    ]
    if not recall_k_list:
        recall_k_list = [1, 5, 10, 20]
    recall_top = max(recall_k_list)

    retrieval_by_sid: dict[str, list[str]] = {}
    entities_by_sid: dict[str, list[str]] = {}
    sample_id2ans = {}
    if args.multi_step_qa:

        for sample_id in tqdm(unique_sample_ids):
            retrieved_list: list[str] = []
            ents_for_ret: list[str] = []
            try:
                question = id2sample[sample_id]["question"]
                ans = inference_with_db.answer_with_qa_collapsing(
                    question,
                    sample_id,
                    use_qualifiers=use_qualifiers,
                    use_filtered_triplets=use_filtered_triplets,
                )
                sample_id2ans[sample_id] = ans
                if isinstance(ans, int):
                    ans = str(ans)
                linked = inference_with_db.identify_relevant_entities_from_question_with_llm(
                    question=question,
                    sample_id=sample_id,
                    use_entity_types=args.structured_inference,
                )
                st = inference_with_db.get_1_hop_supporting_triplets(
                    list(dict.fromkeys(linked or [])),
                    sample_id,
                    use_qualifiers,
                    use_filtered_triplets,
                )
                ents = _entities_for_passage_retrieval(linked, st)
                ents_for_ret = ents
                retrieved_list = _retrieve_passages_from_entity_edges(
                    triplets_db,
                    ents,
                    sample_id if sample_scoped else None,
                    recall_top,
                )
                with jsonlines.open(
                    f"qa_logs/{triplets_db_name}_{model_tag}_structured_{args.structured_inference}_multi_step_{args.multi_step_qa}_use_qualifiers_{args.use_qualifiers}_use_filtered_triplets_{args.use_filtered_triplets}_hotpot_test_run_{str(args.run_number)}.jsonl",
                    "a",
                ) as f:
                    f.write({"sample_id": sample_id, "answer": ans})
                logger.info("-" * 100)
                logger.info(
                    sample_id,
                    " | ",
                    question,
                    " | ",
                    ans,
                    " | ",
                    normalize(ans),
                    " | ",
                    id2sample[sample_id]["answer"],
                    " | ",
                    normalize(id2sample[sample_id]["answer"]),
                )
            except Exception:
                logger.exception(
                    "Error for sample_id=%s, question=%s", sample_id, question
                )
            retrieval_by_sid[sample_id] = retrieved_list
            entities_by_sid[sample_id] = ents_for_ret

    else:
        for sample_id in tqdm(unique_sample_ids):
            retrieved_list: list[str] = []
            ents_for_ret: list[str] = []
            try:
                question = id2sample[sample_id]["question"]
                identified_entities = (
                    inference_with_db.identify_relevant_entities_from_question_with_llm(
                        question=question,
                        sample_id=sample_id,
                        use_entity_types=args.structured_inference,
                    )
                )

                if not identified_entities:
                    logger.error(
                        f"[DEBUG] No entities identified for sample_id={sample_id}, question={question}"
                    )
                supporting_triplets, ans = inference_with_db.answer_question_with_llm(
                    question=question,
                    linked_entities=identified_entities,
                    sample_id=sample_id,
                    use_qualifiers=use_qualifiers,
                    use_filtered_triplets=use_filtered_triplets,
                )

                ents = _entities_for_passage_retrieval(
                    identified_entities, supporting_triplets
                )
                ents_for_ret = ents
                retrieved_list = _retrieve_passages_from_entity_edges(
                    triplets_db,
                    ents,
                    sample_id if sample_scoped else None,
                    recall_top,
                )

                logger.info(
                    sample_id,
                    " | ",
                    question,
                    " | ",
                    ans,
                    " | ",
                    normalize(ans),
                    " | ",
                    id2sample[sample_id]["answer"],
                    " | ",
                    normalize(id2sample[sample_id]["answer"]),
                )
                logger.info("-" * 100)
                sample_id2ans[sample_id] = ans
                with jsonlines.open(
                    f"qa_logs/{triplets_db_name}_{model_tag}_structured_{args.structured_inference}_multi_step_{args.multi_step_qa}_use_qualifiers_{args.use_qualifiers}_use_filtered_triplets_{args.use_filtered_triplets}_hotpot_test_run_{str(args.run_number)}.jsonl",
                    "a",
                ) as f:
                    f.write({"sample_id": sample_id, "answer": ans})
            except Exception:
                logger.exception(
                    "Error for sample_id=%s, question=%s", sample_id, question
                )
            retrieval_by_sid[sample_id] = retrieved_list
            entities_by_sid[sample_id] = ents_for_ret

    # --- Aggregate QA metrics (script previously only wrote jsonl) ---
    eval_ids = [sid for sid in unique_sample_ids if sid in id2sample]
    gold_answers = []
    predicted_answers = []
    for sid in eval_ids:
        raw = sample_id2ans.get(sid)
        if raw is None:
            predicted_answers.append("")
        else:
            predicted_answers.append(str(raw).strip())
        gold_answers.append([id2sample[sid]["answer"]])

    print(f"\n--- QA summary ---")
    print(f"Questions in DB with dataset match: {len(eval_ids)}")
    print(f"Predictions stored: {len(sample_id2ans)}")
    print(f"Failed / skipped (empty prediction in metrics): {len(eval_ids) - len(sample_id2ans)}")

    try:
        from hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score

        em, _ = QAExactMatch().calculate_metric_scores(
            gold_answers=gold_answers, predicted_answers=predicted_answers
        )
        f1, _ = QAF1Score().calculate_metric_scores(
            gold_answers=gold_answers, predicted_answers=predicted_answers
        )
        out = {**em, **f1}
        label = "HippoRAG MRQA normalize (comparable to run_wikontic_baseline)"
    except ImportError:
        out = _qa_metrics_local(gold_answers, predicted_answers)
        label = "local normalize (install HippoRAG src for paper-style metrics)"

    print(f"\nQA metrics ({label}):")
    for k, v in out.items():
        print(f"  {k}: {v}")
    print(f"Per-question answers: qa_logs/…_hotpot_test_run_{args.run_number}.jsonl")

    gold_docs_by_query = [
        _gold_passage_texts_hotpot_item(id2sample[sid], corpus_mode) for sid in eval_ids
    ]
    npass = triplets_db.get_collection("passages").estimated_document_count()
    nedge = triplets_db.get_collection("passage_entity_edges").estimated_document_count()
    use_mongo_passages = npass > 0

    retrieved_docs_by_query: list[list[str]] = []
    for sid in eval_ids:
        if use_mongo_passages:
            retrieved_docs_by_query.append(retrieval_by_sid.get(sid, []))
        else:
            retrieved_docs_by_query.append(
                _retrieve_passages_hotpot_context_overlap(
                    id2sample[sid],
                    entities_by_sid.get(sid, []),
                    recall_top,
                )
            )

    try:
        from hipporag.evaluation.retrieval_eval import RetrievalRecall

        rmetrics, _ = RetrievalRecall(global_config=None).calculate_metric_scores(
            gold_docs=gold_docs_by_query,
            retrieved_docs=retrieved_docs_by_query,
            k_list=recall_k_list,
        )
        if use_mongo_passages:
            rlabel = (
                f"Mongo passage_entity_edges (passages={npass}, edges={nedge}, "
                f"CORPUS_MODE={corpus_mode})"
            )
        else:
            rlabel = (
                f"Hotpot context overlap from Wikontic entities (no `passages` in DB; "
                f"CORPUS_MODE={corpus_mode}) — for HippoRAG/PPR-style recall use hybrid build"
            )
        print(f"\nRetrieval metrics — {rlabel}:")
        for name, val in sorted(rmetrics.items()):
            print(f"  {name}: {val}")
    except ImportError:
        print(
            "\nRetrieval Recall@k: add HippoRAG/src to PYTHONPATH (RetrievalRecall)."
        )
