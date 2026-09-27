# Compass architecture

## Role

Compass is a **ranker**, not an agent. The host (Admeasy, Cursor, LangGraph, etc.) still owns tool execution, OAuth, and approvals.

## Scoring

- Feature text: `utterance | dept | pin | prior | channel | tool | desc`
- Tokenizer: unigrams + bigrams → hashed into EmbeddingBag buckets
- MLP → sigmoid score per candidate
- Sort descending; optional `no_tool` / `meta_first` policy

## Family conflict (v1)

If utterance matches `social_post` or `social_dm` and lacks mail words, downweight tools whose name/desc match `mail_send`. Boost matching social tools slightly.

## Dynamic candidates

Production passes 100–1000 candidates. Training samples gold + hard negatives from **other families** into pools of ~20–80 so the model never memorizes a fixed global list.

## Serving

- Local: `serving/infer_torch.py`
- Optional: SageMaker serverless endpoint wrapping the same code
