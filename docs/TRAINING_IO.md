# Training IO

## Trajectory JSONL (one example)

```json
{
  "id": "ex1",
  "user_utterance": "post reel",
  "state": {"department": "marketing", "pin": null, "prior_tools": [], "channel": "web"},
  "candidates": [{"name": "instagram_publish_reel", "score_hint": 0.95}, {"name": "zoho_send_mail", "score_hint": 0.1}],
  "positive": {"tool": "instagram_publish_reel", "args": {}},
  "negatives": ["zoho_send_mail"],
  "no_tool": false,
  "labels": {"family": "social_post", "intent": "publish_reel"}
}
```

## Ranker rows

Flattened pairwise / listwise rows for HashBag training: positive vs in-batch negatives from the same dynamic pool.

## Quality gates

- gold ∈ candidates
- pin consistency when pin set
- CRM family split (native / Zoho / SFDC)
- social vs mail conflict: social gold must not be a mail tool
- duplicate rate

## Metrics

Recall@1 / Recall@3 vs keyword baseline; invalid-tool rate; pin-respect rate; family confusion matrix.
