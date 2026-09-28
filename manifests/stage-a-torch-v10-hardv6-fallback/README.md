# Stage A Torch v10 hardv6 fallback

This is the newest completed SageMaker checkpoint deployed to the Compass public demo.

- Training job: `jev-v10-hardv6-fallback-g4dn12xlarge-20260928-154101`
- Dataset: 600,000 hard-negative rows, 28 epochs
- Best matched recall@1 at k=8: **0.7183476395**
- S3 artifact: `s3://amazon-sagemaker-530448593594-us-east-1-bewonmqz0j9mp3/jev-router/models/stage-a-torch-v10-hardv6-fallback/model.tar.gz`
- Live endpoint: `jev-router-ranker-v1`

The binary checkpoint is kept in SageMaker S3 rather than committed to Git. The public Vercel site proxies the live endpoint through the AWS try-me Lambda.
