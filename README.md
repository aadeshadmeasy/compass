# Compass

**Fast, lightweight ranking for dynamic MCP tool routing.**

Plain-language intent → ranked tools.

Compass is a compact MCP tool ranker that selects relevant tools from dynamic candidate sets of tens to thousands of tools. It uses a learned HashBag scoring model to keep routing local, fast, and inexpensive.

It is **a ranker, not an agent loop.**

[Live Demo](https://compass-self-nu.vercel.app) · [GitHub](https://github.com/aadeshadmeasy/compass) · [Architecture](docs/ARCHITECTURE.md) · [Training I/O](docs/TRAINING_IO.md)

---

## Overview

| | |
|:--|:--|
| **Model** | `stage-a-torch-v10-hardv6-fallback` |
| **Retrieval quality** | R@1@k8 ≈ 0.7183 |
| **Hard-negative training data** | 600,000 rows |
| **Enriched tool catalog** | ~32,000 tools |
| **Serving** | AWS SageMaker Serverless |
| **Public demo** | AWS Lambda Function URL |

*Status: 28 September 2026. Metrics reflect the current reported checkpoint.*

## Try it

**Live ranking API**

```bash
curl -sS -X POST \
  https://2zzcfljnazt226l7ogaku2ek3a0zlkyi.lambda-url.us-east-1.on.aws/api/rank \
  -H 'Content-Type: application/json' \
  -d '{
    "utterance": "post reel",
    "department": "marketing",
    "top_k": 5
  }'
```

Health check:

```bash
curl -sS \
  https://2zzcfljnazt226l7ogaku2ek3a0zlkyi.lambda-url.us-east-1.on.aws/health
```

Explore the [interactive demo](https://compass-self-nu.vercel.app).

The demo is a temporary public endpoint. It uses the `stage-a-torch-v10-hardv6-fallback` checkpoint and does not require AWS credentials in this repository.

## Why Compass

MCP workspaces can accumulate hundreds or thousands of tools across email, social media, analytics, CRM, and commerce platforms.

Routing every request through a large LLM adds latency and cost. It can also confuse tools from unrelated domains.

For example, **“post reel” should rank social publishing tools above email-sending tools.**

Compass addresses this with lightweight scoring and family-conflict handling.

## How it works

```text
Natural-language request
          │
          ▼
Optional lexical retrieval
          │
          ▼
Dynamic candidate set
          │
          ▼
Learned HashBag scoring
          │
          ▼
Tool-family conflict policy
          │
          ▼
Ranked top-k tools
```

Each request supplies a state and a candidate set. Compass scores the available candidates and returns the highest-ranked tools.

Training uses dynamic candidate pools containing the correct tool and hard negatives from other tool families, rather than a fixed global tool list.

## Routing examples

| Request | Expected ranking |
|:--|:--|
| `post reel` | Instagram, Buffer, Ayrshare |
| `dm reply` | Instagram, WhatsApp, Slack messaging |
| `GA4 summary` | GA4 analytics tools |
| `send email` | Gmail, Zoho Mail, CRM email tools |

The goal is to improve tool selection without introducing an LLM decision loop into every routing request.

## Quickstart

**Requirements:** Python 3 and a compatible PyTorch installation.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install torch flask
```

Add `boto3` only if your setup uses the SageMaker endpoint.

Start the local demo:

```bash
python demo/compass_demo_app.py
```

Open `http://127.0.0.1:8765`.

The local demo requires the appropriate sample catalog and checkpoint files to be available.

## Model & serving

| Component | Details |
|:--|:--|
| Current checkpoint | `stage-a-torch-v10-hardv6-fallback` |
| Training data | 600k hard-negative rows |
| Serving endpoint | `jev-router-ranker-v1` |
| Serving platform | SageMaker Serverless |
| Public API | AWS Lambda Function URL |
| Checkpoint storage | Private S3 bucket |

**Model artifacts**

- [Checkpoint manifest](manifests/stage-a-torch-v10-hardv6-fallback/README.md)
- Training and inference details: [Architecture](docs/ARCHITECTURE.md)

Model weights (`.pt` and `.tar.gz`) are excluded from Git. The full training catalog and serving artifacts remain in private S3 storage.

A separate GPU training job is active but is not part of the currently deployed checkpoint.

## Repository

```text
compass/
├── demo/           Local Flask demo
├── serving/        HashBag inference and ranking
├── scripts/        Training, data minting, quality gates
├── data/           Public sample datasets
├── docs/           Architecture and training specifications
├── manifests/      Checkpoint documentation
└── site/           Public demo interface
```

The repository includes a stratified sample of approximately 10,000 tools and approximately 2,000 public routing trajectories. The full training catalog remains private.

## Research notes

- **Dynamic candidates:** Rank the tools supplied for the current request.
- **Hard negatives:** Train against confusing alternatives from other tool families.
- **Lightweight inference:** Avoid an LLM agent loop for routine tool ranking.
- **Family-conflict handling:** Reduce cross-domain mistakes, such as selecting email tools for social publishing.

See [Training I/O](docs/TRAINING_IO.md) for the training data contract.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Credits

Built for Admeasy AI routing research.

**Compass** is the public project name. *Jev* was the internal codename.
