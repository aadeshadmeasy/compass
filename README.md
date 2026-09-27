# Compass

**Compass** is a tiny, fast **MCP tool ranker**: plain English → ranked tools from a dynamic candidate set (tens to thousands of MCP tools).

It is **not** an LLM agent loop. Compass learns to *rank* whatever candidates you provide (HashBag), so routing stays sub-millisecond locally and cheap on serverless.

> Public GitHub: [aadeshadmeasy/compass](https://github.com/aadeshadmeasy/compass)

## Why

MCP workspaces easily accumulate 100–1000+ tools across Gmail, Zoho Mail, Instagram, Slack, GA4, Shopify, Salesforce, Buffer/Ayrshare, etc. Asking a big LLM to pick every turn is slow and conflates **mail** with **social** (e.g. “post reel” should never rank `zoho_send_mail`).

Compass:

1. Takes `state` (utterance, optional department/pin) + `candidates[]`
2. Scores each candidate with a compact HashBag
3. Applies light **family conflict** policy (social vs mail) at score time
4. Returns top-k tools with scores

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch flask  # + boto3 only if using SageMaker endpoint

# Local demo (loads sample catalog + checkpoint if present)
python demo/compass_demo_app.py
# open http://127.0.0.1:8765
```

Try:

| Utterance | Expect |
|-----------|--------|
| `post reel` | Instagram / Buffer / Ayrshare publish — **not** Zoho/Gmail |
| `dm reply` | IG / WhatsApp / Slack DM tools — **not** mail send |
| `GA4 summary` | `ga4_*` analytics |
| `send email…` | Gmail / Zoho Mail / CRM mail |

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/TRAINING_IO.md](docs/TRAINING_IO.md).

```
utterance → (optional lexical retrieve) → candidates[]
        → HashBag score → family conflict → top-k
```

Training uses **dynamic candidate pools** per example (gold + hard negatives from other families), never a fixed global tool list.

## Repo layout

```
demo/           try-me Flask UI
serving/        infer_torch HashBag load + rank
scripts/        train / mint / quality gate (sanitized)
data/           sample catalog subset (public)
docs/           architecture + training IO
manifests/      placeholder for checkpoints (large weights via Releases/S3)
```

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Credits

Built for Admeasy Ai routing research. Brand: **Compass**. Internal codename was “Jev”.

## Status (2026-09-28)

- Catalog path: ~32k tools. This repo ships a **sample** catalog only (`data/tool_catalog_sample.jsonl`), not the full ~32k list.
- Sarvam mint is **in progress** under a 50M token cap.
- Live endpoint: `jev-router-ranker-v1` on **v4** (stage-a-torch-v4). v5 is not deployed.
- GPU SageMaker train is **deferred** until quota case `179054010300961` is approved (expected tomorrow).
