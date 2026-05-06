# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thai recipe assistant backend — a RAG + LLM service that answers cooking questions with recipe recommendations, nutritional analysis, and personality-driven responses in Thai, English, and Simplified Chinese.

## Running & Testing

```bash
# Run the server (default port 8000, or set PORT env var)
python main.py

# Offline retrieval testing (no LLM, no API key needed)
python test.py "your query"
python test.py "your query" --json
python test.py "your query" --k_children 40 --top_parents 3 --alts 2
```

**Required env var:** `GEMINI_API_KEY` — set in `.env` or environment.

No test suite or linter is configured.

## Architecture

### Data Flow

```
User Query → Thai spell-correct (pythainlp) → Embed (BAAI/bge-m3)
  → FAISS search (k=40 child chunks)
  → Deduplicate + re-rank by parent, weighted by field
  → Top 3 parent menus
  → Parse nutrition → Evaluate against WHO daily limits
  → Build prompt with persona/language/tone + nutrition context
  → Gemini 2.0-Flash → Response + nutrition_analysis JSON
```

### Key Files

| File | Role |
|------|------|
| `main.py` | Everything: FastAPI app, RAG retrieval, nutrition parsing/evaluation, Gemini LLM integration |
| `ollama.py` | Alternative backend using local Ollama (Gemma3) instead of Gemini |
| `test.py` | CLI tool for offline retrieval testing |
| `new-vector-database.py` | Data ingestion pipeline — rebuilds FAISS index from CSV |
| `create-index.py` | Simpler index builder (older approach) |

### Vector Index (Hierarchical RAG)

- **Parent:** full menu records in `menus_docstore.jsonl` (one JSON per line, keyed by `parent_id`)
- **Children:** field-level text chunks in `child_texts.pkl` + `child_meta.pkl` — each chunk tagged with its field type
- **Index:** `thai_recipes_bge_m3.index` — FAISS `IndexFlatIP` with L2-normalized vectors (cosine similarity)
- **Field weights at re-rank:** ingredients 1.5× > method 1.1× > name 1.0× > nutrition 0.9× > tags 0.5×

### Nutrition Analysis

Regex-parses `kcal`, `protein_g`, `fat_g`, `carb_g`, `sugar_g`, `sodium_mg` from retrieved menu text, then evaluates each against hardcoded WHO daily caps (energy 2000 kcal, fat ≤65 g, sugar ≤50 g, sodium ≤2000 mg, etc.). Results returned alongside the LLM answer.

### Personality Profiles

Three personas controlled by the `personality` request field:
- `souschef` — polite cooking teacher, uses "ฉัน"
- `buddy` — casual friend, uses "ฉัน"
- `chef-ian` — formal Masterchef, uses "ผม" + "ครับ"

Language (`lang`) accepts `th`, `en`, `cn`/`zh` — controls the LLM system prompt language and tone map.

### API

**`POST /ask`**
```json
// Request
{ "question": "...", "personality": "souschef", "lang": "th" }

// Response
{
  "answer": "...",
  "context_n_menus": 3,
  "nutrition_analysis": [{ "parent_id", "menu_name", "nutrition_parsed", "nutrition_eval" }],
  "guideline_used": {}
}
```

Session state (chat history) is tracked per `X-Session-ID` request header using Gemini's Chat API.

CORS is open to `http://localhost:5173` and `https://jirawatchn.github.io`.
