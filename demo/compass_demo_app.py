#!/usr/bin/env python3
"""Compass try-me demo — plain English → ranked MCP tools.

Run on Grok Bot box:
  cd .. && .venv/bin/python demo/jev_demo_app.py
Open http://127.0.0.1:8765 (watch via Grok Bot computer) or public tunnel if started.
Prefers LOCAL HashBag (fast). Set JEV_DEMO_BACKEND=endpoint to use SageMaker.
"""
from __future__ import annotations

import json
import re
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serving"))

from flask import Flask, jsonify, request, Response

from infer_torch import load_model, rank, enrich_candidates, load_catalog, feature_text, score_texts

PORT = int(os.environ.get("JEV_DEMO_PORT", "8765"))
BACKEND = os.environ.get("JEV_DEMO_BACKEND", "local")  # local | endpoint
ENDPOINT = os.environ.get("JEV_ENDPOINT", "compass-ranker-v1")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
MODEL_DIR = ROOT / "manifests" / "stage-a-torch-v3"
CATALOG_PATH = ROOT / "data" / "tool_catalog_sample.jsonl"
TOP_N_FALLBACK = 80

app = Flask(__name__)

model, cfg = load_model(MODEL_DIR)
catalog = load_catalog(CATALOG_PATH)
ALL_TOOLS: list[dict] = []
with CATALOG_PATH.open() as f:
    for line in f:
        ALL_TOOLS.append(json.loads(line))


def slim(t: dict) -> dict:
    name = t.get("name") or t.get("federated_name")
    return {
        "name": name,
        "description": (t.get("description") or "")[:200],
        "department": t.get("department") or "platform",
        "risk_level": t.get("risk_level") or "low",
        "server_slug": t.get("server_slug") or "native",
    }


# Meta / fallback tools only — NEVER force email into every candidate set.
META_ALWAYS = ["discover_capabilities", "no_op"]

# Seed tools by family — injected ONLY when utterance/pin matches that family.
EMAIL_FAMILY_SEEDS = [
    "gmail_send_email", "mail_compose_email", "crm_send_email", "crm_compose_lead_email",
    "crm_draft_lead_email", "crm_send_lead_email", "GMAIL_SEND_EMAIL", "GMAIL_CREATE_EMAIL_DRAFT",
    "gmail_list_messages", "zoho_mail_send_email", "zoho_mail_send", "zoho_send_mail",
    "outlook_send_email", "marketing_send_bulk_email", "zoho_mail_list_recent", "zoho_mail_search",
]
SOCIAL_POST_SEEDS = [
    "ig_graph_media_publish", "instagram_get_recent_posts", "ayrshare_post", "ayrshare_schedule",
    "ayrshare_post_video", "buffer_create_post", "buffer_post", "buffer_mcp_schedule_post",
    "threads_create_post", "facebook_create_page_post", "TWITTER_CREATION_OF_A_POST",
    "marketing_autopilot",
]
SOCIAL_DM_SEEDS = [
    "instagram_list_dms", "instagram_send_dm", "instagram_get_conversation_messages",
    "instagram_get_message", "instagram_list_pending_inbound", "ig_graph_conversations",
    "whatsapp_send_message", "whatsapp_list_recent_messages", "post_message", "get_thread",
]
GA4_SEEDS = ["ga4_summary", "ga4_acquisition", "ga4_top_pages", "ga4_conversions", "run_report"]

EMAIL_WORDS = (
    "email", "e-mail", "gmail", "compose", "proposal",
    "outlook", "zoho mail", "send mail", "draft mail", "mailbox", "inbox email",
)
SOCIAL_POST_WORDS = (
    "reel", "reels", "story", "stories", "post", "posts", "publish", "schedule post",
    "instagram", "ig ", " ig", "threads", "facebook", "twitter", "x.com", "reddit",
    "buffer", "ayrshare", "social", "feed", "carousel", "caption",
)
SOCIAL_DM_WORDS = (
    "dm", "dms", "direct message", "direct msg", "inbox dm", "reply dm",
    "whatsapp", "wa reply", "slack dm", "slack message", "message reply",
)
GA4_WORDS = ("ga4", "analytics", "traffic", "kpi", "sessions", "conversions report")
MAIL_PIN_HINTS = ("google-workspace", "gmail", "zoho-mail", "zoho_mail", "outlook")

MAIL_NAME_RE = ("mail", "gmail", "email", "outlook", "smtp")
SOCIAL_POST_NAME_RE = (
    "ayrshare", "buffer", "instagram", "ig_graph_media", "threads_create",
    "facebook_create", "twitter", "reddit", "publish", "schedule_post",
)
SOCIAL_DM_NAME_RE = (
    "instagram_send_dm", "instagram_list_dm", "instagram_get_conversation",
    "instagram_get_message", "instagram_list_pending", "ig_graph_conversation",
    "whatsapp", "slack", "post_message", "get_thread",
)


def _utt_has(utt: str, words: tuple[str, ...]) -> bool:
    return any(w in utt for w in words)


_WORD_DM_RE = re.compile(r"(?<![a-z0-9])(dms?|direct[\s-]?messages?)(?![a-z0-9])", re.I)
_WORD_IG_RE = re.compile(r"(?<![a-z0-9])(ig|instagram)(?![a-z0-9])", re.I)


def detect_families(utterance: str, pin: str | None) -> set[str]:
    utt = (utterance or "").lower()
    pin_l = (pin or "").lower()
    fams: set[str] = set()
    mailish = _utt_has(utt, EMAIL_WORDS) or "email" in utt or "gmail" in utt or "outlook" in utt
    if re.search(r"(?<![a-z])mails?(?![a-z])", utt):
        mailish = True
    # "inbox" is mail only without IG/DM social signals
    if "inbox" in utt and not _WORD_DM_RE.search(utt) and not _WORD_IG_RE.search(utt) and "instagram" not in utt:
        mailish = True
    if mailish or any(h in pin_l for h in MAIL_PIN_HINTS):
        fams.add("mail_send")
    if _utt_has(utt, SOCIAL_POST_WORDS) or _WORD_IG_RE.search(utt):
        if any(x in utt for x in ("reel", "reels", "story", "stories", "post", "posts", "publish", "schedule", "buffer", "ayrshare", "threads", "facebook", "twitter", "reddit", "x.com", "social", "caption", "carousel", "feed")):
            fams.add("social_post")
    if _WORD_DM_RE.search(utt) or _utt_has(utt, ("direct message", "whatsapp", "wa reply", "slack dm", "slack message", "message reply")):
        if "mail_send" not in fams:
            fams.add("social_dm")
    if any(x in utt for x in ("reel", "reels")) and "mail" not in utt:
        fams.add("social_post")
    if _utt_has(utt, GA4_WORDS):
        fams.add("ga4")
    # Admeasy + email/send → mail (never treat substring "dm" inside admeasy as social_dm)
    if "admeasy" in utt and "social_post" not in fams and "social_dm" not in fams:
        if _utt_has(utt, EMAIL_WORDS) or ("send" in utt and "post" not in utt):
            fams.add("mail_send")
    return fams


def tool_family_tags(name: str, desc: str = "") -> set[str]:
    blob = f"{name} {desc}".lower()
    tags: set[str] = set()
    if any(x in blob for x in MAIL_NAME_RE):
        tags.add("mail_send")
    if any(x in blob for x in SOCIAL_POST_NAME_RE):
        tags.add("social_post")
    if any(x in blob for x in SOCIAL_DM_NAME_RE):
        tags.add("social_dm")
    if "ga4" in blob or blob.startswith("run_report"):
        tags.add("ga4")
    return tags


def apply_family_conflict(scores: list[tuple[str, float]], utterance: str, pin: str | None, by_cand: dict[str, dict]) -> list[tuple[str, float]]:
    """Downweight mail tools when utterance is social_post/social_dm without mail words."""
    fams = detect_families(utterance, pin)
    has_mail = "mail_send" in fams
    social = fams & {"social_post", "social_dm"}
    out = []
    for name, sc in scores:
        tags = tool_family_tags(name, (by_cand.get(name) or {}).get("description") or "")
        if social and not has_mail:
            if "mail_send" in tags and not (tags & social):
                sc = sc * 0.15  # hard downweight conflicting mail
            elif social & tags:
                sc = min(1.0, sc * 1.35 + 0.05)  # boost matching social family
        out.append((name, float(sc)))
    out.sort(key=lambda x: -x[1])
    return out


def select_candidates(utterance: str, department: str | None, pin: str | None, limit: int = 50) -> list[dict]:
    """Dynamic lexical + family-tag retrieval. Never pad with unrelated mail tools."""
    utt = (utterance or "").lower()
    fams = detect_families(utterance, pin)
    kws = [w for w in utt.replace("'", " ").replace("-", " ").split() if len(w) > 1]
    # strong social / DM / post keyword boosts for retrieval
    extra: list[str] = []
    if "mail_send" in fams:
        extra.extend(["email", "gmail", "mail", "compose", "send", "crm", "zoho_mail", "inbox", "proposal"])
    if "social_post" in fams:
        extra.extend([
            "reel", "reels", "post", "story", "stories", "instagram", "ig", "threads",
            "facebook", "twitter", "reddit", "buffer", "ayrshare", "publish", "schedule",
            "media", "caption", "social",
        ])
    if "social_dm" in fams:
        extra.extend([
            "dm", "dms", "instagram", "ig", "whatsapp", "slack", "message", "conversation",
            "reply", "inbox", "thread",
        ])
    if "ga4" in fams:
        extra.extend(["ga4", "analytics", "semrush", "search_console", "traffic", "kpi", "sessions"])
    if "admeasy" in utt and "mail_send" in fams:
        extra.extend(["crm", "gmail", "email", "mail"])
    kws = list(dict.fromkeys(kws + extra))

    by_name = {(t.get("name") or t.get("federated_name")): t for t in ALL_TOOLS}
    picked: dict[str, dict] = {}

    # Meta only (not email)
    for n in META_ALWAYS:
        if n in by_name:
            picked[n] = slim(by_name[n])

    # Family seeds — conditional
    seed_lists: list[list[str]] = []
    if "mail_send" in fams:
        seed_lists.append(EMAIL_FAMILY_SEEDS)
    if "social_post" in fams:
        seed_lists.append(SOCIAL_POST_SEEDS)
    if "social_dm" in fams:
        seed_lists.append(SOCIAL_DM_SEEDS)
    if "ga4" in fams:
        seed_lists.append(GA4_SEEDS)
    for seeds in seed_lists:
        for n in seeds:
            if n in by_name and n not in picked:
                picked[n] = slim(by_name[n])

    scored = []
    for t in ALL_TOOLS:
        name = t.get("name") or t.get("federated_name") or ""
        blob = " ".join(
            str(t.get(k) or "") for k in ("name", "federated_name", "description", "when_to_use", "department", "server_slug")
        ).lower()
        hits = 0
        if department and (t.get("department") or "").lower() == department.lower():
            hits += 2
        hits += sum(1 for k in kws if k in blob)
        # family tag bonus
        tags = tool_family_tags(name, t.get("description") or "")
        if fams & tags:
            hits += 3
        # hard: if social intent without mail, do not prefer mail tools in retrieval
        if (fams & {"social_post", "social_dm"}) and "mail_send" not in fams and "mail_send" in tags:
            hits = max(0, hits - 5)
            if hits == 0:
                continue
        if hits:
            scored.append((hits, t))
    scored.sort(key=lambda x: (-x[0], x[1].get("name") or ""))
    for _, t in scored:
        name = t.get("name") or t.get("federated_name")
        if name not in picked:
            picked[name] = slim(t)
        if len(picked) >= limit:
            break

    # Short utterances: still retrieve by lexical+family; pad with SAME-family neighbors, never unrelated mail
    if len(picked) < 15:
        prefer_tags = fams or set()
        for t in ALL_TOOLS:
            name = t.get("name") or t.get("federated_name")
            if name in picked:
                continue
            tags = tool_family_tags(name, t.get("description") or "")
            if prefer_tags and not (prefer_tags & tags):
                # allow meta / neutral platform tools only
                if (t.get("department") or "") not in ("platform", "ops", ""):
                    continue
            if (fams & {"social_post", "social_dm"}) and "mail_send" not in fams and "mail_send" in tags:
                continue
            picked[name] = slim(t)
            if len(picked) >= min(limit, 25):
                break

    cands = list(picked.values())
    if pin:
        slug_u = pin.replace("-", "_")
        filtered = [
            c for c in cands
            if c["name"].startswith(f"ext.{slug_u}.") or c.get("server_slug") == pin
            or c["name"] in ("discover_capabilities", "no_op")
        ]
        if filtered:
            cands = filtered
    return enrich_candidates(cands, catalog)


def invoke_endpoint(payload: dict) -> tuple[dict, float]:
    import boto3
    smr = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "sagemaker"), region_name=REGION).client("sagemaker-runtime")
    t0 = time.perf_counter()
    resp = smr.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    out = json.loads(resp["Body"].read())
    return out, (time.perf_counter() - t0) * 1000.0


def apply_conflict_to_ranked(out: dict, payload: dict) -> dict:
    """Re-score ranked list with family conflict policy (social vs mail)."""
    state = payload.get("state") or {}
    utterance = state.get("user_utterance") or ""
    pin = state.get("pin")
    by_cand = {(c.get("name") or c.get("federated_name")): c for c in payload.get("candidates") or []}
    ranked = list(out.get("ranked") or [])
    if not ranked:
        return out
    pairs = [(r["tool"], float(r["score"])) for r in ranked]
    # also re-score full candidate set for better top-k after downweight
    all_pairs = []
    # Prefer rebuilding from model scores already in ranked; if top_k small, expand via candidates scores in out
    # Re-run conflict on the returned ranked only first; caller may pass fuller list
    pairs2 = apply_family_conflict(pairs, utterance, pin, by_cand)
    # If conflict changed order, rebuild ranked preserving metadata
    meta = {r["tool"]: r for r in ranked}
    new_ranked = []
    for name, sc in pairs2:
        r = dict(meta[name])
        r["score"] = float(sc)
        new_ranked.append(r)
    out = dict(out)
    out["ranked"] = new_ranked
    out["policy_version"] = (out.get("policy_version") or cfg.get("policy_version") or "jev") + "+fam_conflict_v1"
    return out


def local_predict(payload: dict) -> tuple[dict, dict]:
    n_buckets = cfg["n_buckets"]
    t0 = time.perf_counter()
    # Score ALL candidates then apply family conflict before top_k trim
    state = payload.get("state") or {}
    cands = payload.get("candidates") or []
    top_k = int(payload.get("top_k") or 5)
    texts = [feature_text(state, c) for c in cands]
    raw_scores = score_texts(model, texts, n_buckets) if texts else []
    by_cand = {(c.get("name") or c.get("federated_name")): c for c in cands}
    pairs = []
    for c, s in zip(cands, raw_scores):
        name = c.get("name") or c.get("federated_name")
        pairs.append((name, float(s)))
    pairs = apply_family_conflict(pairs, state.get("user_utterance") or "", state.get("pin"), by_cand)
    ranked_raw = []
    for name, sc in pairs:
        c = by_cand.get(name) or {"name": name}
        risk = (c.get("risk_level") or "").lower()
        ranked_raw.append({
            "tool": name,
            "score": float(sc),
            "args_sketch": {},
            "meta": name in ("discover_capabilities", "workspace_tools_snapshot", "oauth_status", "tool_schema_get"),
            "needs_approval": risk in ("high", "critical"),
        })
    # no_tool / meta_first via shared rank() for policy parity on chitchat
    out = rank(payload, model, n_buckets)
    # Replace ranked with conflict-aware full ranking
    no_tool = out.get("no_tool")
    meta_first = False
    if no_tool:
        ranked = out.get("ranked") or ranked_raw[:top_k]
    else:
        ranked = ranked_raw[:top_k]
        tau = 0.25
        best = ranked[0]["score"] if ranked else 0.0
        if best < tau:
            disc = next((r for r in ranked_raw if r["tool"] == "discover_capabilities"), None)
            if disc and disc["tool"] != (ranked[0]["tool"] if ranked else None):
                ranked = [disc] + [r for r in ranked if r["tool"] != "discover_capabilities"]
                ranked = ranked[:top_k]
                meta_first = True
    e2e_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    score_texts(model, texts[: min(8, len(texts))] or ["x"], n_buckets)
    fwd_us = (time.perf_counter() - t1) * 1e6
    out = {
        "ranked": ranked,
        "no_tool": bool(no_tool),
        "meta_first": bool(meta_first),
        "deterministic": True,
        "policy_version": (cfg.get("policy_version") or "jev-stage-a-v3-clean") + "+fam_conflict_v1",
    }
    return out, {
        "backend": "local",
        "e2e_ms": round(e2e_ms, 3),
        "pure_forward_us": round(fwd_us, 2),
        "n_candidates": len(cands),
        "families": sorted(detect_families(state.get("user_utterance") or "", state.get("pin"))),
        "note": "pure_forward_us is HashBag only; e2e_ms includes feature build + family conflict policy",
    }


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Compass — try me</title>
<style>
  :root { --bg:#0b1220; --card:#141e33; --fg:#e8eefc; --muted:#9bb0d0; --acc:#5b8cff; --ok:#3dd68c; --bad:#ff6b7a; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background: radial-gradient(1200px 600px at 10% -10%, #1a2a55, var(--bg)); color: var(--fg); min-height:100vh; }
  main { max-width: 860px; margin: 0 auto; padding: 28px 18px 60px; }
  h1 { font-size: 1.55rem; margin: 0 0 6px; }
  .sub { color: var(--muted); margin-bottom: 18px; line-height: 1.45; }
  .card { background: var(--card); border: 1px solid #243353; border-radius: 14px; padding: 16px; margin-bottom: 14px; }
  label { display:block; font-size: 0.85rem; color: var(--muted); margin-bottom: 6px; }
  textarea, input, select { width:100%; background:#0d1628; color:var(--fg); border:1px solid #2a3c63;
    border-radius:10px; padding:10px 12px; font-size:1rem; }
  textarea { min-height: 88px; resize: vertical; }
  .row { display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px; margin-top:10px; }
  button { margin-top: 12px; background: var(--acc); color:white; border:0; border-radius:10px;
           padding: 11px 16px; font-weight:600; font-size:1rem; cursor:pointer; width:100%; }
  button:disabled { opacity:0.6; cursor:wait; }
  .hint { font-size:0.8rem; color:var(--muted); margin-top:8px; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
  .chip { background:#1a2744; border:1px solid #2e426e; color:var(--fg); border-radius:999px;
          padding:6px 12px; font-size:0.8rem; cursor:pointer; }
  table { width:100%; border-collapse: collapse; }
  th, td { text-align:left; padding:8px 6px; border-bottom:1px solid #243353; font-size:0.95rem; }
  th { color:var(--muted); font-weight:500; }
  .score { font-variant-numeric: tabular-nums; color: var(--ok); }
  .meta { color:var(--muted); font-size:0.85rem; margin-top:10px; line-height:1.4; }
  .warn { color:#ffd166; }
  code { background:#0d1628; padding:1px 6px; border-radius:6px; }
</style>
</head>
<body>
<main>
  <h1>Compass — live tool ranker</h1>
  <p class="sub">Type plain English. This demos <b>prompt → ranked MCP tools</b> (not SageMaker Studio deploy notebooks).
  Default backend is <b>local HashBag</b> on the Grok Bot box (sub-ms). Endpoint RTT is tens–hundreds of ms.</p>

  <div class="card">
    <label for="utt">Prompt</label>
    <textarea id="utt" placeholder="I'm using Admeasy and I want to send an email"></textarea>
    <div class="row">
      <div>
        <label for="dept">Department (optional)</label>
        <input id="dept" placeholder="crm / marketing / sales"/>
      </div>
      <div>
        <label for="pin">Pin / server slug (optional)</label>
        <input id="pin" placeholder="e.g. google-workspace-mcp"/>
      </div>
      <div>
        <label for="backend">Backend</label>
        <select id="backend">
          <option value="local" selected>local (fast)</option>
          <option value="endpoint">SageMaker endpoint</option>
        </select>
      </div>
    </div>
    <div class="chips">
      <span class="chip" data-u="I'm using Admeasy and I want to send an email" data-d="crm">Admeasy email</span>
      <span class="chip" data-u="Send an email with Gmail to the client about the proposal" data-d="crm">Gmail proposal</span>
      <span class="chip" data-u="Show me last 28 days GA4 summary please" data-d="marketing">GA4 summary</span>
      <span class="chip" data-u="post reel" data-d="marketing">post reel</span>
      <span class="chip" data-u="dm reply" data-d="marketing">dm reply</span>
      <span class="chip" data-u="check Instagram inbox DMs" data-d="marketing">IG DMs</span>
      <span class="chip" data-u="thanks!" data-d="">thanks (no_tool)</span>
    </div>
    <button id="go">Rank tools</button>
    <p class="hint">Ignore Studio notebooks named <code>*-deploy-*</code> — those only create endpoints.</p>
  </div>

  <div class="card" id="out">
    <div class="meta">Results appear here.</div>
  </div>
</main>
<script>
const utt = document.getElementById('utt');
const dept = document.getElementById('dept');
const pin = document.getElementById('pin');
const backend = document.getElementById('backend');
const go = document.getElementById('go');
const out = document.getElementById('out');
document.querySelectorAll('.chip').forEach(c => c.onclick = () => {
  utt.value = c.dataset.u; dept.value = c.dataset.d || '';
});
go.onclick = async () => {
  go.disabled = true; out.innerHTML = '<div class="meta">Running…</div>';
  try {
    const r = await fetch('/api/rank', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ utterance: utt.value, department: dept.value || null, pin: pin.value || null, backend: backend.value, top_k: 5 })
    });
    const j = await r.json();
    if (j.error) { out.innerHTML = `<div class="meta warn">${j.error}</div>`; return; }
    const rows = (j.ranked||[]).map((x,i)=>`<tr><td>${i+1}</td><td><code>${x.tool}</code>${x.needs_approval?' <span class="warn">approval</span>':''}${x.meta?' <span class="warn">meta</span>':''}</td><td class="score">${(x.score*100).toFixed(2)}%</td></tr>`).join('');
    out.innerHTML = `
      <table><thead><tr><th>#</th><th>Tool</th><th>Score</th></tr></thead><tbody>${rows||'<tr><td colspan=3>none</td></tr>'}</tbody></table>
      <div class="meta">
        utterance: <b>${(j.utterance||'').replace(/</g,'&lt;')}</b><br/>
        backend: <code>${j.timing?.backend}</code> · candidates: ${j.timing?.n_candidates}
        · e2e: <b>${j.timing?.e2e_ms ?? j.timing?.rtt_ms} ms</b>
        ${j.timing?.pure_forward_us!=null?` · pure forward: <b>${j.timing.pure_forward_us} µs</b>`:''}
        ${j.no_tool?' · <span class="warn">no_tool hint</span>':''}
        ${j.meta_first?' · <span class="warn">meta_first</span>':''}<br/>
        policy: ${j.policy_version||''}
        ${j.timing?.note?`<br/><span class="warn">${j.timing.note}</span>`:''}
      </div>`;
  } catch (e) {
    out.innerHTML = `<div class="meta warn">${e}</div>`;
  } finally { go.disabled = false; }
};
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.post("/api/rank")
def api_rank():
    body = request.get_json(force=True, silent=True) or {}
    utterance = (body.get("utterance") or body.get("user_utterance") or "").strip()
    if not utterance:
        return jsonify({"error": "utterance required"}), 400
    department = body.get("department") or None
    pin = body.get("pin") or None
    top_k = int(body.get("top_k") or 5)
    backend = (body.get("backend") or BACKEND).lower()
    cands = select_candidates(utterance, department, pin)
    payload = {
        "state": {
            "user_utterance": utterance,
            "department": department,
            "pin": pin,
            "prior_tools": body.get("prior_tools") or [],
            "channel": body.get("channel") or "web",
        },
        "candidates": cands,
        "top_k": top_k,
    }
    try:
        if backend == "endpoint":
            out, rtt = invoke_endpoint(payload)
            timing = {
                "backend": "endpoint",
                "rtt_ms": round(rtt, 2),
                "e2e_ms": round(rtt, 2),
                "n_candidates": len(cands),
                "note": "SageMaker serverless InvokeEndpoint wall-clock RTT (not model-only).",
            }
        else:
            out, timing = local_predict(payload)
        return jsonify({
            "utterance": utterance,
            "ranked": out.get("ranked"),
            "no_tool": out.get("no_tool"),
            "meta_first": out.get("meta_first"),
            "policy_version": out.get("policy_version") or cfg.get("policy_version"),
            "timing": timing,
            "candidate_sample": [c["name"] for c in cands[:12]],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "backend_default": BACKEND,
        "endpoint": ENDPOINT,
        "model": str(MODEL_DIR),
        "n_tools": len(ALL_TOOLS),
        "policy_version": cfg.get("policy_version"),
    })


if __name__ == "__main__":
    print(f"Compass demo on http://127.0.0.1:{PORT}  backend={BACKEND}  tools={len(ALL_TOOLS)}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
