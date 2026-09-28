# stage-a-torch-v5 (box interim)

**GPU train deferred until quota** for preferred `ml.g4dn.xlarge` (case `179054010300961` / request `ecee886646f74b26b6e725e723e812b49hRuW34F`).

## Box interim metrics (not yet serving)

| Metric | Value |
|--------|-------|
| val AUC | 0.9582 |
| val recall@1 | 0.8493 |
| val recall@3 | 0.9483 |

This interim checkpoint is historical and is not the current serving stage; production now uses **stage-a-torch-v10-hardv6-fallback**.

S3 interim: `s3://amazon-sagemaker-530448593594-us-east-1-bewonmqz0j9mp3/jev-router/models/stage-a-torch-v5/model.tar.gz`
