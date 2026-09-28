# Compass try-me implementation

`try_me_lambda_handler.py` is the public AWS Lambda Function URL handler used by the live demo. It proxies ranking to SageMaker endpoint `jev-router-ranker-v1`, serves the try-me page, and includes the `noop_gate_v2` guard so action requests cannot be misclassified as `no_op`.

The live endpoint uses the production `stage-a-torch-v4` weights. The Vercel site in `../site/` remains a static UI whose `/api/rank` and `/health` rewrites target that Lambda Function URL.
