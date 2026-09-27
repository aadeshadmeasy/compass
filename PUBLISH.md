# Publishing Compass to GitHub

Target repo: **https://github.com/aadeshadmeasy/Compass** (create empty repo first; do not force-push over unrelated history).

## 1. Create empty GitHub repo

```bash
gh repo create aadeshadmeasy/Compass --public --description "Compass — fast HashBag MCP tool ranker (plain English → ranked tools)" --clone=false
```

Or create via GitHub UI (empty, no README/license — this tree already has them).

## 2. Push this sanitized tree

```bash
cd /workspace/jev-router/oss
git init
git add .
git status   # confirm no .env, no keys, no private Admeasy logs
git commit -m "Initial public release: Compass MCP tool ranker"
git branch -M main
git remote add origin git@github.com:aadeshadmeasy/Compass.git
# or: https://github.com/aadeshadmeasy/Compass.git
git push -u origin main
```

## 3. Sanity checklist before push

- [ ] No `SARVAM_API_KEY`, AWS keys, or `.env` files
- [ ] No private Admeasy federated logs / customer utterances beyond synthetic samples
- [ ] `data/tool_catalog_sample.jsonl` is a **subset** only
- [ ] Demo screenshots (if any) redacted
- [ ] LICENSE is Apache-2.0
- [ ] README brands **Compass** and links the GitHub repo

## 4. Optional: model weights

Large `model.pt` / `model.tar.gz` → GitHub Release or public S3, linked from README. Do not commit multi‑MB binaries unless intentional.

## 5. Demo screenshots note

Place PNGs under `docs/screenshots/` (post reel / dm reply ranking). Prefer synthetic prompts only.
