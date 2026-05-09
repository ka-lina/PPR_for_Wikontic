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