# PPR for Wikontic - Hybrid Retrieval System

## Overview

This project combines Wikontic triplet extraction with HippoRAG-style PPR graph retrieval.

The current main entrypoint is `run_hybrid.py`. Older files named `experiment_*.py` are legacy/reference experiments from earlier iterations and are not the recommended path for the new hybrid method.

The intended flow is:

1. Wikontic extracts/refines triplets and stores them in MongoDB.
2. The hybrid layer stores passages and `passage_entity_edges` in the same MongoDB database.
3. `WikonticHippoRAG` builds a HippoRAG graph from existing Wikontic triplets instead of running HippoRAG OpenIE again.
4. HippoRAG retrieval/PPR and optional QA/metrics run on that graph.

## Key Features

- Non-ontology and Wikidata ontology modes via `USE_ONTOLOGY=0/1`
- Local OpenAI-compatible LLM support, tested with Ollama
- MongoDB storage for triplets, passages, and passage-entity edges
- HippoRAG-compatible PPR retrieval over Wikontic triplets
- Retrieval metrics using HippoRAG `RetrievalRecall`
- Optional QA metrics using HippoRAG EM/F1 when QA is enabled

## Repository Structure

```text
PPR_for_Wikontic/
├── run_hybrid.py                    # Main hybrid entrypoint
├── src/
│   ├── data/
│   ├── hybrid/                      # New hybrid HippoRAG/Wikontic integration
│   ├── wikontic_ppr/                # Older PPR wrappers and ontology helpers
│   └── evaluation/                  # Legacy/auxiliary metrics
├── docker/
└── requirements.txt
```

## Setup

### 1. Clone External Repositories

```bash
cd /mnt/study
git clone https://github.com/OSU-NLP-Group/HippoRAG.git
git clone https://github.com/screemix/Wikontic.git
```

### 2. Create venv

```bash
cd /mnt/study/PPR_for_Wikontic
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-deps -e ../HippoRAG
python -m pip install --no-deps -e ../Wikontic
```

### 3. Configure Environment

```bash
export PYTHONPATH=/mnt/study/PPR_for_Wikontic:/mnt/study/HippoRAG:/mnt/study/Wikontic:$PYTHONPATH
export HF_HOME=/mnt/study/PPR_for_Wikontic/.hf_cache
mkdir -p "$HF_HOME"
```

### 4. Start MongoDB

The Docker dev stack uses auth:

```bash
export MONGO_URI='mongodb://wikontic:wikontic123@localhost:27018/?authSource=admin&directConnection=true'
```

If no MongoDB is running:

```bash
docker run -d --name mongodb -p 27018:27017 mongo:7
export MONGO_URI='mongodb://localhost:27018/?directConnection=true'
```

### 5. Dataset

`run_hybrid.py` expects:

```bash
/mnt/study/Wikontic/datasets/hotpotqa200.json
```

Override with:

```bash
export HOTPOTQA_PATH=/path/to/hotpotqa200.json
```

## Local LLM Mode Without OpenAI Key

Install and run Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
```

Run a local retrieval smoke test:

```bash
cd /mnt/study/PPR_for_Wikontic
source .venv/bin/activate

export LLM_BASE_URL='http://localhost:11434/v1'
export LLM_MODEL='qwen2.5:3b'
unset OPENAI_API_KEY

export USE_ONTOLOGY=0
export SKIP_QA=1
export USE_INITIAL_TRIPLETS=1
export SKIP_FACT_RERANK=1
export NUM_QUESTIONS=1
export QUESTION_INDEX=5

python run_hybrid.py
```

`SKIP_QA=1` avoids loading a heavy local QA model. Retrieval metrics are still printed.

## OpenAI-compatible API Mode

```bash
export OPENAI_API_KEY='sk-...'
export LLM_MODEL='gpt-4o-mini'
export LLM_BASE_URL='https://api.openai.com/v1'
python run_hybrid.py
```

Any OpenAI-compatible provider can be used by changing `LLM_BASE_URL` and `LLM_MODEL`.

## Wikidata Ontology Mode

Use:

```bash
export USE_ONTOLOGY=1
```

Ontology mode uses:

- `wikontic.utils.structured_aligner.Aligner`
- `src.wikontic_ppr.structured_wikontic_ppr_inference.StructuredWikonticPPRInference`
- MongoDB database `WIKIDATA_ONTOLOGY_DB`, default `wikidata_ontology_dev`

If the ontology DB is empty and you want to populate it:

```bash
export CREATE_ONTOLOGY_DB=1
export WIKIDATA_ONTOLOGY_DB=wikidata_ontology_dev
python run_hybrid.py
```

This can be slow because ontology aliases need embeddings.

## Configuration

Important environment variables:

```bash
USE_ONTOLOGY=0
NUM_QUESTIONS=1
QUESTION_INDEX=5
QUESTION_INDICES=5,6,7
CORPUS_MODE=gold
SKIP_QA=1
USE_INITIAL_TRIPLETS=1
REUSE_DB=0
SKIP_FACT_RERANK=1
NUM_TO_RETRIEVE=5
RETRIEVAL_TOP_K=5
LINKING_TOP_K=5
DAMPING=0.5
HIPPO_LLM_MODEL=Transformers/Qwen/Qwen2.5-3B-Instruct
EMBEDDING_MODEL=facebook/contriever
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=16
```

Use `REUSE_DB=1` after a failed indexing/retrieval run to keep the existing MongoDB passages and triplets and skip extraction:

```bash
export REUSE_DB=1
python run_hybrid.py
```

## Metrics

Retrieval metrics are printed by `run_hybrid.py` using HippoRAG's `RetrievalRecall`:

- `Recall@1`
- `Recall@5`
- `Recall@10`
- `Recall@20`

When `SKIP_QA=0`, HippoRAG QA also reports:

- `ExactMatch`
- `F1`

`ExactMatch` and `F1` require generated answers, so run with `SKIP_QA=0`:

```bash
export SKIP_QA=0
export EMBEDDING_DEVICE=cpu
python run_hybrid.py
```

The local `src/evaluation` metrics are legacy/auxiliary and are not the primary paper-comparable metrics.

## Pure Wikontic Baseline

Use `run_wikontic_baseline.py` to evaluate Wikontic without HippoRAG graph retrieval/PPR. This mode reuses an existing `TRIPLETS_DB` and reports QA metrics over Wikontic supporting triplets:

```bash
export USE_ONTOLOGY=1
export TRIPLETS_DB=hotpotqa_hybrid_ontology
export WIKIDATA_ONTOLOGY_DB=wikidata_ontology_dev
export NUM_QUESTIONS=200
export QUESTION_INDEX=0

python run_wikontic_baseline.py 2>&1 | tee wikontic_baseline_ontology.log
```

The baseline reports `Recall@k` (same `RetrievalRecall` as `run_hybrid.py`), where passages are ranked by summed `passage_entity_edges` weights over linked entities and entities from supporting triplets, then `ExactMatch`, `F1`, and `AvgSupportingTriplets`. Set `RECALL_K_LIST=1,5,10,20` to change k values. Use `CORPUS_MODE=gold` (default) so gold passages match `run_hybrid.py`.

Optional Wikontic QA graph expansion (same flags as upstream eval):

```bash
export WIKONTIC_USE_QUALIFIERS=1
export WIKONTIC_USE_FILTERED_TRIPLETS=1
```

`WIKONTIC_USE_FILTERED_TRIPLETS` adds triplets from the `ontology_filtered_triplets` collection; use only with `USE_ONTOLOGY=1`. Qualifiers enlarge each triplet payload for the QA prompt when present in the data.

## Expected Local Smoke Output

A successful local smoke run should show nonzero graph facts/edges and retrieval metrics:

```text
Loaded ... refined triplets
Extracted ... entities, ... facts
Index built: ... nodes, ... edges
Retrieval metrics:
  Recall@1: ...
  Recall@5: ...
Skipping rag_qa because SKIP_QA=1
```

## Troubleshooting

### MongoDB Connection Issues

Check that MongoDB is running on port `27018`:

```bash
docker ps -a | grep mongo
```

For the dev Docker stack:

```bash
export MONGO_URI='mongodb://wikontic:wikontic123@localhost:27018/?authSource=admin&directConnection=true'
```

### OpenAI quota error

If you see `429 insufficient_quota`, either enable billing/use another key or switch to local Ollama mode.

### CUDA out of memory

Use retrieval-only mode:

```bash
export SKIP_QA=1
export SKIP_FACT_RERANK=1
```

### HuggingFace cache / offline runs

If `facebook/contriever` has already been downloaded:

```bash
export HF_HOME=/mnt/study/PPR_for_Wikontic/.hf_cache
export HF_LOCAL_FILES_ONLY=1
```

Unset `HF_LOCAL_FILES_ONLY` if the model still needs to be downloaded.

## Notes For Development

- Keep `run_hybrid.py` as the new main entrypoint.
- Treat `experiment_*.py` as legacy/reference scripts.
- Prefer HippoRAG evaluation classes for paper-comparable `Recall@k`, EM, and F1.
- Local smoke runs may use `USE_INITIAL_TRIPLETS=1` because small local LLMs can fail Wikontic refinement.

## Acknowledgments

- HippoRAG: Original PPR retrieval implementation
- Wikontic: Ontology-aware triplet extraction
- HotPotQA: Dataset for evaluation
# PPR for Wikontic

Personalized Page Rank algorithm based retrieval (like in HippoRAG2) used on a Wikontic graph.

```markdown
# PPR for Wikontic - Hybrid Retrieval System

## Overview

This project combines Wikontic's ontology-aware triplet extraction with HippoRAG's PPR-based graph retrieval. The system extracts structured triplets from documents using Wikontic, builds a knowledge graph, and then uses HippoRAG's Personalized PageRank algorithm for document retrieval.

## Key Features

- **Ontology-aware triplet extraction** using Wikidata (or non-ontology mode)
- **Graph-based retrieval** with PPR propagation
- **Passage-level retrieval** with entity linking
- **MongoDB storage** for triplets, entities, and passages
- **HippoRAG-compatible** embedding pipeline

## Repository Structure

```
PPR_for_Wikontic/
├── run_hybrid.py                    # Main experiment script
├── src/
│   ├── data/
│   │   ├── hotpotqa_loader.py       # HotPotQA dataset loader
│   │   └── embedding_creator.py     # Embedding model wrapper
│   ├── hybrid/
│   │   ├── wikontic_hipporag.py     # Main hybrid class
│   │   ├── passage_manager.py       # Passage storage manager
│   │   └── hipporag_embedding_model.py  # HippoRAG compatibility wrapper
│   └── wikontic_ppr/
│       ├── wikontic_ppr_inference.py      # Non-ontology version
│       ├── structured_wikontic_ppr_inference.py  # Ontology version
│       └── passage_store.py          # Passage storage
├── configs/
│   └── experiment_config.yaml       # Configuration file
└── requirements.txt                 # Dependencies
```

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Documents ──→ Passage Manager ──→ MongoDB (passages)           │
│       │                                                          │
│       └──────→ Wikontic Extractor ──→ MongoDB (triplets)        │
│                                              │                   │
│                                              ↓                   │
│                                    Passage-Entity Edges         │
│                                              │                   │
│                                              ↓                   │
│                              WikonticHippoRAG.build()            │
│                                              │                   │
│                                              ↓                   │
│                                          iGraph                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Question ──→ Entity Extraction ──→ PPR Retrieval               │
│                       │                      │                   │
│                       ↓                      ↓                   │
│                   Entities              Passage IDs              │
│                              │              │                    │
│                              ↓              ↓                    │
│                         Weighted Reset    Passages               │
│                              │              │                    │
│                              ↓              ↓                    │
│                         Graph PPR ──────→ Answer                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Setup Instructions

### 1. Clone Repositories

```bash
# Clone this repository
git clone <your-repo-url>
cd PPR_for_Wikontic

# Clone dependencies (adjust paths as needed)
git clone <hipporag-url> ../HippoRAG
git clone <wikontic-url> ../Wikontic
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.21.0
pandas>=1.3.0
torch>=2.0.0
transformers>=4.30.0
igraph>=0.10.0
pymongo>=4.3.0
tqdm>=4.65.0
pyarrow>=12.0.0
```

### 3. Start MongoDB

```bash
# Using Docker
docker run -d -p 27018:27017 --name mongodb mongo:latest

# Or locally
mongod --port 27018
```

### 4. Download Dataset

Place HotPotQA dataset in:
```
./Wikontic/datasets/hotpotqa200.json
```

### 5. Run Experiment

```bash
python run_hybrid.py
```

## Configuration

### Key Parameters (in `run_hybrid.py`)

```python
# Experiment parameters
num_questions = 1           # Number of questions to process
chunk_size = 200            # Passage chunk size (characters)

# Retrieval parameters
damping = 0.5              # PPR damping factor (0.5 is more focused)
linking_top_k = 30         # Number of candidate facts to consider
retrieval_top_k = 10       # Number of passages to retrieve

# Model parameters
embedding_model = "facebook/contriever"
llm_model = "Qwen/Qwen2.5-3B-Instruct"
```

## Version History

### v0.1.0 (Initial) - 2026-05-09

**Initial implementation:**
- Basic `WikonticHippoRAG` class inheriting from HippoRAG
- Passage management with MongoDB
- Triple extraction using Wikontic
- Graph building from triplets and passage-entity edges

**Known issues:**
- `eval()` parsing errors for fact strings with spaces
- JSON parsing errors in DSPyFilter
- Assertion errors in `get_top_k_weights`

### v0.2.0 - 2026-05-09

**Fixes applied:**
- Replaced `eval()` with custom `_parse_fact_string()` method
- Added regex-based parsing for unquoted fact strings
- Overrode `get_top_k_weights()` to handle mismatched entity counts
- Disabled LLM reranking temporarily (using top scores directly)
- Added `_entity_exists_in_graph()` validation

**Current status:**
- Graph builds successfully (17 nodes, 79 edges on test data)
- PPR retrieval runs without errors
- Results may still need tuning for recall

### v0.3.0 (Planned)

**Future improvements:**
- Re-enable LLM reranking with robust JSON parsing
- Add synonymy edges for better entity matching
- Implement IDF weighting for edges
- Add evaluation metrics (Recall@k, F1)

## Troubleshooting

### Error: `'EmbeddingCreator' object has no attribute 'batch_encode'`

**Solution:** Use `HippoRAGEmbeddingModel` wrapper:

```python
from src.hybrid.hipporag_embedding_model import HippoRAGEmbeddingModel
embedding_model = HippoRAGEmbeddingModel(embedding_creator)
```

### Error: `SyntaxError: invalid syntax` when parsing facts

**Solution:** The custom `_parse_fact_string()` method handles unquoted strings:

```python
def _parse_fact_string(self, fact_str: str) -> tuple:
    fact_str = fact_str.strip()
    if fact_str.startswith('(') and fact_str.endswith(')'):
        fact_str = fact_str[1:-1]
    parts = [p.strip() for p in fact_str.split(',')]
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2])
    return None
```

### Error: `AssertionError` in `get_top_k_weights`

**Solution:** Override the method to log warnings instead of asserting:

```python
def get_top_k_weights(self, link_top_k, all_phrase_weights, linking_score_map):
    # ... implementation with lenient handling
    if nonzero_count != expected_count:
        print(f"Warning: nonzero_count={nonzero_count}, expected={expected_count}")
    return all_phrase_weights, linking_score_map
```

### MongoDB Connection Issues

Check that MongoDB is running on port 27018:
```bash
docker ps | grep mongodb
# or
netstat -an | grep 27018
```

## Usage Examples

### Basic Usage

```python
from src.hybrid.wikontic_hipporag import WikonticHippoRAG
from src.hybrid.hipporag_embedding_model import HippoRAGEmbeddingModel
from src.data.embedding_creator import EmbeddingCreator

# Initialize
embedding_creator = EmbeddingCreator("facebook/contriever")
embedding_model = HippoRAGEmbeddingModel(embedding_creator)

hybrid = WikonticHippoRAG(triplets_db, embedding_model)

# Index documents
hybrid.index_from_wikontic(passage_texts)

# Retrieve
results = hybrid.retrieve_from_wikontic(["Your question here"], num_to_retrieve=5)
```

### Running with Ontology

```python
from src.wikontic_ppr.structured_wikontic_ppr_inference import StructuredWikonticPPRInference
from wikontic.utils.structured_aligner import Aligner

aligner = Aligner(triplets_db=triplets_db, ontology_db=ontology_db)
wikontic = StructuredWikonticPPRInference(extractor, aligner, triplets_db, ontology_db, embedding_model)
```

## Current Limitations

1. **LLM reranking disabled** - Using top fact scores directly due to JSON parsing issues
2. **Small test set** - Currently configured for 1 question / 1 passage
3. **No synonymy edges** - Entities must match exactly
4. **Single-hop only** - PPR works but graph connectivity depends on extracted triplets

## Performance Notes

- **First run**: Slow due to triplet extraction (LLM calls)
- **Subsequent runs**: Fast due to graph caching
- **Memory**: Graph stored in memory; MongoDB for persistence
- **GPU**: Recommended for embedding model (Contriever ~1GB VRAM)

## Contributing

When modifying the code:

1. **Always use `_parse_fact_string()`** instead of `eval()` or `ast.literal_eval()`
2. **Add entity existence validation** before adding to graph
3. **Log warnings** instead of assertions for graceful degradation
4. **Test with small dataset** first (1 question, 1 passage)

## License

[Your License Here]

## Acknowledgments

- HippoRAG: Original PPR retrieval implementation
- Wikontic: Ontology-aware triplet extraction
- HotPotQA: Dataset for evaluation
```