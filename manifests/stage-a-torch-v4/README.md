# stage-a-torch-v4 (historical serving weights)

**Status:** Historical production stage; superseded in production by `stage-a-torch-v10-hardv6-fallback`.

## Metrics (box train, quality-gated clean rows)

| Metric | Value |
|--------|-------|
| val AUC | 0.9719 |
| val recall@1 | 0.8570 |
| val recall@3 | 0.9705 |
| emb_dim / hidden | 128 / 256 |
| policy | `jev-stage-a-v4-clean+marketplace` |

## Artifacts

Large `model.pt` / serve `model.tar.gz` are **not** in git (see root `.gitignore`).

- S3 model: `s3://amazon-sagemaker-530448593594-us-east-1-bewonmqz0j9mp3/jev-router/models/stage-a-torch-v4/model.tar.gz`
- Unpacked manifests: `s3://…/jev-router/manifests/stage-a-torch-v4/`
- Model Registry: `jev-router-ranker` version 4

Local demo can load weights from S3 or a local `model.pt` dropped into this folder.
