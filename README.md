# Compass

**Compass** is a tiny, fast **MCP tool ranker**: plain English → ranked tools from a dynamic candidate set (tens to thousands of MCP tools).

It is **not** an LLM agent loop. Compass learns to *rank* whatever candidates you provide (HashBag), so routing stays sub-millisecond locally and cheap on serverless.

> Public GitHub: [aadeshadmeasy/compass](https://github.com/aadeshadmeasy/compass)

## Status (2026-09-27 IST)

| Item | State |
|------|--------|
| **Current weights stage** | **`stage-a-torch-v4`** (clean + marketplace; R@1≈0.857) |
| Serving | SageMaker serverless endpoint `jev-router-ranker-v1` |
| Catalog (private/box) | ~32,000 enriched tools |
| Sample in this repo | `data/tool_catalog_sample.jsonl` (~400 tools) |
| **GPU train** | **Deferred until quota** — prefer `ml.g4dn.xlarge` training (case `179054010300961` / request `ecee8866…`). Approved already: `ml.g5.xlarge`, `ml.g5.2xlarge`, `ml.g4dn.2xlarge`. Big GPU `stage-a-torch-v5` starts **tomorrow after approval**. Overnight GPU jobs stopped. |
| v5 data | Long-utterance Sarvam mint + dynamic candidates (under 50M token hard cap) |

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

## Weights

`*.pt` / `*.tar.gz` are gitignored. Current serving weights:

- Stage: **`stage-a-torch-v4`** — see [manifests/stage-a-torch-v4/README.md](manifests/stage-a-torch-v4/README.md)
- S3: `s3://amazon-sagemaker-530448593594-us-east-1-bewonmqz0j9mp3/jev-router/models/stage-a-torch-v4/model.tar.gz`

v5 GPU train is pending quota (see Status table).

## Repo layout

```
demo/           try-me Flask UI
serving/        infer_torch HashBag load + rank
scripts/        train / mint / quality gate (sanitized)
data/           sample catalog subset (public)
docs/           architecture + training IO
manifests/      stage READMEs (large weights via S3 / Releases)
```

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Credits

Built for Admeasy Ai routing research. Brand: **Compass**. Internal codename was “Jev”.
