"""Temporary public Jev ranker demo — Lambda Function URL → SageMaker endpoint.

GET / or /index.html → try-me HTML
POST /api/rank → {utterance, pin?, department?, top_k?} → live InvokeEndpoint
OPTIONS /* → CORS
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import boto3

ENDPOINT = os.environ.get("JEV_ENDPOINT", "jev-router-ranker-v1")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
POOL_LIMIT = int(os.environ.get("JEV_POOL_LIMIT", "80"))
CATALOG_PATH = Path(__file__).with_name("catalog_demo.json")

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
}

_smr = None
_CATALOG: list[dict] | None = None
_BY_NAME: dict[str, dict] | None = None

# META_ALWAYS removed — see meta_tools_for_utterance / wants_no_op
EMAIL_FAMILY_SEEDS = [
    "gmail_send_email", "mail_compose_email", "crm_send_email", "crm_compose_lead_email",
    "crm_draft_lead_email", "crm_send_lead_email", "GMAIL_SEND_EMAIL", "GMAIL_CREATE_EMAIL_DRAFT",
    "gmail_list_messages", "zoho_mail_send_email", "zoho_mail_send", "zoho_send_mail",
    "outlook_send_email", "marketing_send_bulk_email", "zoho_mail_list_recent", "zoho_mail_search",
]
SOCIAL_POST_SEEDS = [
    "instagram_publish_reel", "instagram_publish_story", "instagram_publish_photo",
    "instagram_publish_media", "instagram_publish_carousel", "buffer_schedule_reel",
    "ayrshare_post_reel", "ayrshare_post_video", "facebook_publish_reel",
    "tiktok_upload_video", "tiktok_publish_video", "autoposting_create_reel",
    "ig_graph_media_publish", "instagram_get_recent_posts", "ayrshare_post", "ayrshare_schedule",
    "buffer_create_post", "buffer_post", "buffer_mcp_schedule_post",
    "threads_create_post", "facebook_create_page_post", "TWITTER_CREATION_OF_A_POST",
    "marketing_autopilot", "ext.instagram_mcp.instagram_publish_reel",
    "ext.buffer_mcp.buffer_schedule_reel", "ext.ayrshare_mcp.ayrshare_post_reel",
    "ayrshare_upload_media", "files_upload",
]
SOCIAL_DM_SEEDS = [
    "instagram_list_dms", "instagram_send_dm", "instagram_get_conversation_messages",
    "instagram_get_message", "instagram_list_pending_inbound", "ig_graph_conversations",
    "whatsapp_send_message", "whatsapp_list_recent_messages", "post_message", "get_thread",
    "ext.instagram_mcp.instagram_reply_dm", "ext.slack_mcp.slack_send_dm",
]
GA4_SEEDS = ["ga4_summary", "ga4_acquisition", "ga4_top_pages", "ga4_conversions", "run_report", "ext.ga4_mcp.run_report"]

EMAIL_WORDS = (
    "email", "e-mail", "gmail", "compose", "proposal",
    "outlook", "zoho mail", "zoho", "send mail", "draft mail", "mailbox", "inbox email",
    "bhej", "bhejo", "bhejni", "bhejna",  # Hinglish send
)
SOCIAL_POST_WORDS = (
    "reel", "reels", "story", "stories", "post", "posts", "publish", "schedule post",
    "instagram", "insta", "ig ", " ig", "threads", "facebook", "twitter", "x.com", "reddit",
    "buffer", "ayrshare", "social", "feed", "carousel", "caption",
    "video", "videos", "upload", "banani", "banana", "banao",  # Hinglish make + video/upload
)
GA4_WORDS = ("ga4", "analytics", "traffic", "kpi", "sessions", "conversions report")
MAIL_PIN_HINTS = ("google-workspace", "gmail", "zoho-mail", "zoho_mail", "outlook")
MAIL_NAME_RE = ("mail", "gmail", "email", "outlook", "smtp")
SOCIAL_POST_NAME_RE = (
    "ayrshare", "buffer", "instagram", "ig_graph_media", "threads_create",
    "facebook_create", "twitter", "reddit", "publish", "schedule_post", "reel",
)
SOCIAL_DM_NAME_RE = (
    "instagram_send_dm", "instagram_list_dm", "instagram_get_conversation",
    "instagram_get_message", "instagram_list_pending", "ig_graph_conversation",
    "whatsapp", "slack", "post_message", "get_thread", "reply_dm",
)
_WORD_DM_RE = re.compile(r"(?<![a-z0-9])(dms?|direct[\s-]?messages?)(?![a-z0-9])", re.I)
_WORD_IG_RE = re.compile(r"(?<![a-z0-9])(ig|instagram|insta)(?![a-z0-9])", re.I)


# Meta tools: do NOT always inject. no_op only for chitchat / nothing-to-do.
META_DISCOVER = "discover_capabilities"
META_NOOP = "no_op"
_NO_OP_EXACT = {
    "thanks", "thanks!", "thank you", "thank you!", "thx", "ty",
    "ok", "ok.", "okay", "okay.", "ok cool", "cool", "got it",
    "cancel", "cancelled", "canceled", "nevermind", "never mind",
    "no tool", "no tools", "nothing to do", "do nothing", "no thanks",
    "pass", "n/a", "na", "hi", "hello", "hey", "good morning",
    "good afternoon", "good evening", "who are you", "what is jev",
}
_NO_OP_SUBSTR = (
    "thanks", "thank you", "never mind", "nevermind", "nothing to do",
    "do nothing", "no tool", "no thanks", "cancel that", "forget it",
)
# English + Hinglish action / content tokens — NEVER no_op / no_tool when present.
_ACTION_HINT_RE = re.compile(
    r"(?i)(?<![a-z0-9])("
    r"post|posts|publish|schedule|send|compose|draft|reply|check|show|list|get|"
    r"open|create|upload|share|reel|reels|story|stories|video|videos|email|mail|gmail|"
    r"zoho|outlook|instagram|insta|ig|dm|dms|analytics|ga4|report|inbox|caption|"
    # Hinglish action verbs / forms (romanized)
    r"banani|banana|banao|bana|banaye|banau|banaun|"
    r"krni|karni|karna|karo|karun|karke|kar|"
    r"bhej|bhejo|bhejni|bhejna|bhejdo|"
    r"daal|dal|daalo|dalni|dalna|"
    r"bhejne|upload\s*kar|post\s*kar"
    r")(?![a-z0-9])"
)
_STRONG_ACTION_FAMILIES = frozenset({"social_post", "mail_send", "social_dm", "ga4"})


def has_action_tokens(utterance: str) -> bool:
    """True when utterance has English/Hinglish action or content verbs."""
    return bool(_ACTION_HINT_RE.search(utterance or ""))


def wants_no_op(utterance: str) -> bool:
    """True only for thanks/ok/cancel/nevermind/no-tool / nothing-to-do style utts.

    Hinglish/English action verbs (banani/upload/krni/video/reel/post/mail/bhej/daal)
    MUST NEVER return True — even if utterance also has polite filler.
    """
    utt = (utterance or "").lower().strip()
    if not utt:
        return False
    # Hard veto: any action token → never no_op
    if has_action_tokens(utt):
        return False
    if utt in _NO_OP_EXACT:
        return True
    # Short chitchat-only: allow substring match when utterance is short
    if len(utt) <= 40 and any(p in utt for p in _NO_OP_SUBSTR):
        return True
    return False


def is_action_like(utterance: str, pin: str | None = None) -> bool:
    if wants_no_op(utterance):
        return False
    utt = (utterance or "").strip()
    if not utt:
        return False
    fams = detect_families(utterance, pin)
    if fams & _STRONG_ACTION_FAMILIES:
        return True
    if fams:
        return True
    return has_action_tokens(utt)


def meta_tools_for_utterance(utterance: str) -> list[str]:
    """Conditional meta injection — never always-force no_op into every pool."""
    out: list[str] = []
    utt = (utterance or "").lower()
    if wants_no_op(utterance):
        out.append(META_NOOP)
    # discover only for capability / help / unsure inventory questions
    if any(x in utt for x in ("what can you", "capabilities", "what tools", "discover", "help me find", "which tools")):
        out.append(META_DISCOVER)
    return out


def demote_noop_ranked(ranked: list, utterance: str, pin: str | None, top_k: int) -> list:
    """After ranking: if action-like, strip no_op from top_k (drop or push below)."""
    if not ranked:
        return ranked
    if not is_action_like(utterance, pin):
        return ranked
    kept = [r for r in ranked if (r.get("tool") or r.get("name")) != META_NOOP]
    if len(kept) == len(ranked):
        return ranked
    return kept[:top_k]


def _tool_name(r: dict) -> str:
    return (r.get("tool") or r.get("name") or "")


def boost_ranked_for_intent(ranked: list, utterance: str, top_k: int) -> list:
    """Deterministic post-rank nudges for clear lexical intents (dept optional).

    - insta/instagram/ig + reel → prefer instagram_publish_reel (and other *publish_reel*)
    - zoho (+ mail/email) → prefer zoho_* mail tools over delete_email / unrelated
    - video/upload Hinglish → prefer *publish_reel* / *post_video* / *upload_video*
    """
    if not ranked:
        return ranked
    utt = (utterance or "").lower()
    rows = list(ranked)

    def promote_matching(pred, limit: int = 3) -> None:
        nonlocal rows
        hits = [r for r in rows if pred(_tool_name(r).lower())]
        if not hits:
            return
        rest = [r for r in rows if r not in hits]
        rows = hits[:limit] + rest
        # de-dupe preserving order
        seen = set()
        out = []
        for r in rows:
            n = _tool_name(r)
            if n in seen:
                continue
            seen.add(n)
            out.append(r)
        rows = out

    igish = bool(_WORD_IG_RE.search(utt) or "instagram" in utt or "insta" in utt)
    if igish and any(x in utt for x in ("reel", "reels")):
        promote_matching(lambda n: "instagram_publish_reel" in n or n.endswith("instagram_publish_reel"))
        # demote pure twitter create when IG reel requested
        tw = [r for r in rows if "twitter" in _tool_name(r).lower()]
        nontw = [r for r in rows if "twitter" not in _tool_name(r).lower()]
        if nontw:
            rows = nontw + tw

    if "zoho" in utt:
        promote_matching(lambda n: "zoho" in n and any(x in n for x in ("mail", "email", "send")))
        # demote destructive/non-zoho mail ops
        bad = [r for r in rows if any(x in _tool_name(r).lower() for x in ("delete_email", "add_label"))]
        good = [r for r in rows if r not in bad]
        if good:
            rows = good + bad

    if ("video" in utt or "upload" in utt) and any(
        x in utt for x in ("banani", "banana", "banao", "krni", "karni", "karna", "upload", "reel")
    ):
        promote_matching(
            lambda n: any(
                x in n
                for x in (
                    "instagram_publish_reel",
                    "publish_reel",
                    "post_reel",
                    "post_video",
                    "upload_video",
                    "ayrshare_post_video",
                    "files_upload",
                )
            )
        )

    return rows[:top_k]


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Jev Router — AWS try-me (temp)</title>
<style>
  :root { --bg:#0b1220; --card:#141e33; --fg:#e8eefc; --muted:#9bb0d0; --acc:#5b8cff; --ok:#3dd68c; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif;
         background: radial-gradient(1200px 600px at 10% -10%, #1a2a55, var(--bg)); color: var(--fg); min-height:100vh; }
  main { max-width: 860px; margin: 0 auto; padding: 28px 18px 60px; }
  h1 { font-size: 1.55rem; margin: 0 0 6px; }
  .sub { color: var(--muted); margin-bottom: 18px; line-height: 1.45; }
  .card { background: var(--card); border: 1px solid #243353; border-radius: 14px; padding: 16px; margin-bottom: 14px; }
  label { display:block; font-size: 0.85rem; color: var(--muted); margin-bottom: 6px; }
  textarea, input { width:100%; background:#0d1628; color:var(--fg); border:1px solid #2a3c63;
    border-radius:10px; padding:10px 12px; font-size:1rem; }
  textarea { min-height: 88px; resize: vertical; }
  .row { display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:10px; }
  button { margin-top: 12px; background: var(--acc); color:white; border:0; border-radius:10px;
           padding: 11px 16px; font-weight:600; font-size:1rem; cursor:pointer; width:100%; }
  button:disabled { opacity:0.6; cursor:wait; }
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
  .badge { display:inline-block; background:#243353; color:#9bb0d0; border-radius:999px; padding:2px 10px; font-size:0.75rem; margin-left:8px; }
</style>
</head>
<body>
<main>
  <h1>Jev Router — live tool ranker <span class="badge">AWS temp · v4 · noop_gate_v2</span></h1>
  <p class="sub">Plain English → ranked MCP tools via SageMaker endpoint <code>jev-router-ranker-v1</code>
  (stage-a-torch-v4). Temporary public demo — CORS open. Not Cloudflare.</p>
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
    </div>
    <div class="chips">
      <span class="chip" data-u="I'm using Admeasy and I want to send an email" data-d="crm">Admeasy email</span>
      <span class="chip" data-u="Send an email with Gmail to the client about the proposal" data-d="crm">Gmail proposal</span>
      <span class="chip" data-u="Show me last 28 days GA4 summary please" data-d="marketing">GA4 summary</span>
      <span class="chip" data-u="mujhe ek video banani hai phir upload krni h" data-d="">Hinglish video upload</span>
      <span class="chip" data-u="post reel" data-d="marketing">post reel</span>
      <span class="chip" data-u="dm reply" data-d="marketing">dm reply</span>
      <span class="chip" data-u="check Instagram inbox DMs" data-d="marketing">IG DMs</span>
      <span class="chip" data-u="thanks!" data-d="">thanks (no_tool)</span>
    </div>
    <button id="go">Rank tools</button>
  </div>
  <div class="card" id="out"><div class="meta">Results appear here.</div></div>
</main>
<script>
const utt = document.getElementById('utt');
const dept = document.getElementById('dept');
const pin = document.getElementById('pin');
const go = document.getElementById('go');
const out = document.getElementById('out');
const API = (location.pathname.replace(/\/$/, '') || '') + '/api/rank';
document.querySelectorAll('.chip').forEach(c => c.onclick = () => {
  utt.value = c.dataset.u; dept.value = c.dataset.d || '';
});
go.onclick = async () => {
  go.disabled = true; out.innerHTML = '<div class="meta">Invoking SageMaker… (cold start may take a few seconds)</div>';
  try {
    const r = await fetch(API, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ utterance: utt.value, department: dept.value || null, pin: pin.value || null, top_k: 5 })
    });
    const j = await r.json();
    if (j.error) { out.innerHTML = `<div class="meta warn">${j.error}</div>`; return; }
    const rows = (j.ranked||[]).map((x,i)=>`<tr><td>${i+1}</td><td><code>${x.tool}</code>${x.needs_approval?' <span class="warn">approval</span>':''}${x.meta?' <span class="warn">meta</span>':''}</td><td class="score">${(x.score*100).toFixed(2)}%</td></tr>`).join('');
    out.innerHTML = `
      <table><thead><tr><th>#</th><th>Tool</th><th>Score</th></tr></thead><tbody>${rows||'<tr><td colspan=3>none</td></tr>'}</tbody></table>
      <div class="meta">
        utterance: <b>${(j.utterance||'').replace(/</g,'&lt;')}</b><br/>
        backend: <code>sagemaker</code> · endpoint: <code>${j.endpoint||''}</code>
        · candidates: ${j.timing?.n_candidates ?? ''}
        · e2e: <b>${j.timing?.rtt_ms ?? j.latency_ms?.total ?? '?'} ms</b>
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


def _utt_has(utt: str, words: tuple[str, ...]) -> bool:
    return any(w in utt for w in words)


def detect_families(utterance: str, pin: str | None) -> set[str]:
    """Family detect WITHOUT requiring department.

    Hinglish+English: video/upload/reel/instagram/ig/insta → social_post;
    mail/email/zoho/gmail/bhej → mail_send.
    """
    utt = (utterance or "").lower()
    pin_l = (pin or "").lower()
    fams: set[str] = set()
    mailish = (
        _utt_has(utt, EMAIL_WORDS)
        or "email" in utt
        or "gmail" in utt
        or "outlook" in utt
        or "zoho" in utt
    )
    if re.search(r"(?<![a-z])mails?(?![a-z])", utt):
        mailish = True
    if "inbox" in utt and not _WORD_DM_RE.search(utt) and not _WORD_IG_RE.search(utt) and "instagram" not in utt and "insta" not in utt:
        mailish = True
    if mailish or any(h in pin_l for h in MAIL_PIN_HINTS):
        fams.add("mail_send")
    # Social post: English + Hinglish make/upload video/reel/post
    social_keys = (
        "reel", "reels", "story", "stories", "post", "posts", "publish", "schedule",
        "buffer", "ayrshare", "threads", "facebook", "twitter", "reddit", "x.com",
        "social", "caption", "carousel", "feed", "video", "videos", "upload",
        "instagram", "insta", "banani", "banana", "banao", "bana ",
    )
    if _utt_has(utt, SOCIAL_POST_WORDS) or _WORD_IG_RE.search(utt) or any(x in utt for x in social_keys):
        if any(x in utt for x in social_keys) or _WORD_IG_RE.search(utt):
            fams.add("social_post")
    if _WORD_DM_RE.search(utt) or _utt_has(utt, ("direct message", "whatsapp", "wa reply", "slack dm", "slack message", "message reply")):
        if "mail_send" not in fams:
            fams.add("social_dm")
    if any(x in utt for x in ("reel", "reels")) and "mail" not in utt:
        fams.add("social_post")
    # Hinglish "video banani / upload krni" without explicit reel still → social_post
    if ("video" in utt or "upload" in utt) and any(
        x in utt for x in ("banani", "banana", "banao", "bana", "krni", "karni", "karna", "karo", "upload", "post", "reel")
    ):
        fams.add("social_post")
    if _utt_has(utt, GA4_WORDS):
        fams.add("ga4")
    if "admeasy" in utt and "social_post" not in fams and "social_dm" not in fams:
        if _utt_has(utt, EMAIL_WORDS) or ("send" in utt and "post" not in utt):
            fams.add("mail_send")
    # If both mail + social from weak overlap, keep both; SM + demote handle ranking.
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


def slim(t: dict) -> dict:
    name = t.get("name") or t.get("federated_name")
    return {
        "name": name,
        "description": (t.get("description") or "")[:200],
        "department": t.get("department") or "platform",
        "risk_level": t.get("risk_level") or "low",
        "server_slug": t.get("server_slug") or "native",
    }


def load_catalog() -> list[dict]:
    global _CATALOG, _BY_NAME
    if _CATALOG is not None:
        return _CATALOG
    with CATALOG_PATH.open() as f:
        _CATALOG = json.load(f)
    _BY_NAME = {(t.get("name") or t.get("federated_name")): t for t in _CATALOG}
    return _CATALOG


def select_candidates(utterance: str, department: str | None, pin: str | None, limit: int = 80) -> list[dict]:
    tools = load_catalog()
    by_name = _BY_NAME or {}
    utt = (utterance or "").lower()
    fams = detect_families(utterance, pin)
    kws = [w for w in utt.replace("'", " ").replace("-", " ").split() if len(w) > 1]
    extra: list[str] = []
    if "mail_send" in fams:
        extra.extend(["email", "gmail", "mail", "compose", "send", "crm", "zoho_mail", "inbox", "proposal"])
    if "social_post" in fams:
        extra.extend(["reel", "reels", "post", "story", "stories", "instagram", "ig", "insta", "threads", "facebook", "twitter", "reddit", "buffer", "ayrshare", "publish", "schedule", "media", "caption", "social", "video", "upload"])
        # Bias Hinglish video/upload toward reel/publish tools
        if any(x in utt for x in ("video", "upload", "banani", "banana", "reel", "reels")):
            extra.extend(["instagram_publish_reel", "publish_reel", "post_reel", "post_video", "upload_video", "ayrshare_post_video", "tiktok_upload"])
    if "social_dm" in fams:
        extra.extend(["dm", "dms", "instagram", "ig", "whatsapp", "slack", "message", "conversation", "reply", "inbox", "thread"])
    if "ga4" in fams:
        extra.extend(["ga4", "analytics", "semrush", "search_console", "traffic", "kpi", "sessions"])
    if "admeasy" in utt and "mail_send" in fams:
        extra.extend(["crm", "gmail", "email", "mail"])
    kws = list(dict.fromkeys(kws + extra))

    picked: dict[str, dict] = {}
    for n in meta_tools_for_utterance(utterance):
        if n in by_name:
            picked[n] = slim(by_name[n])
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
    for t in tools:
        name = t.get("name") or t.get("federated_name") or ""
        blob = " ".join(str(t.get(k) or "") for k in ("name", "federated_name", "description", "when_to_use", "department", "server_slug")).lower()
        hits = 0
        if department and (t.get("department") or "").lower() == department.lower():
            hits += 2
        hits += sum(1 for k in kws if k in blob)
        tags = tool_family_tags(name, t.get("description") or "")
        if fams & tags:
            hits += 3
        if (fams & {"social_post", "social_dm"}) and "mail_send" not in fams and "mail_send" in tags:
            hits = max(0, hits - 5)
            if hits == 0:
                continue
        if "mail_send" in fams and not (fams & {"social_post", "social_dm"}) and (tags & {"social_post", "social_dm"}):
            continue
        if name == META_NOOP and not wants_no_op(utterance):
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

    if len(picked) < 15:
        prefer_tags = fams or set()
        for t in tools:
            name = t.get("name") or t.get("federated_name")
            if name in picked:
                continue
            # never smuggle no_op into pool unless explicit chitchat
            if name == META_NOOP and not wants_no_op(utterance):
                continue
            tags = tool_family_tags(name, t.get("description") or "")
            if prefer_tags and not (prefer_tags & tags):
                if (t.get("department") or "") not in ("platform", "ops", ""):
                    continue
            if (fams & {"social_post", "social_dm"}) and "mail_send" not in fams and "mail_send" in tags:
                continue
            if "mail_send" in fams and not (fams & {"social_post", "social_dm"}) and (tags & {"social_post", "social_dm"}):
                continue
            picked[name] = slim(t)
            if len(picked) >= min(limit, 25):
                break

    cands = list(picked.values())
    # When intent is GA4-only, keep pool tight so SM is not drowned by marketing noise.
    if fams == {"ga4"}:
        tight = [
            c for c in cands
            if ("ga4" in c["name"].lower() or "ga4" in (c.get("description") or "").lower()
                or c["name"] in ("run_report", "get_metadata", "discover_capabilities") or (c["name"] == "no_op" and wants_no_op(utterance))
                or (c.get("server_slug") or "").startswith("ga4"))
        ]
        if len(tight) >= 5:
            cands = tight
    if pin:
        slug_u = pin.replace("-", "_")
        filtered = [
            c for c in cands
            if c["name"].startswith(f"ext.{slug_u}.") or c.get("server_slug") == pin
            or c["name"] == "discover_capabilities"
            or (c["name"] == "no_op" and wants_no_op(utterance))
        ]
        if filtered:
            cands = filtered
    return cands


def get_smr():
    global _smr
    if _smr is None:
        _smr = boto3.client("sagemaker-runtime", region_name=REGION)
    return _smr


def invoke_endpoint(payload: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    resp = get_smr().invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    out = json.loads(resp["Body"].read())
    return out, (time.perf_counter() - t0) * 1000.0


def _resp(status: int, body: Any, content_type: str = "application/json") -> dict:
    if isinstance(body, (dict, list)):
        body_s = json.dumps(body)
    else:
        body_s = str(body)
    headers = {**CORS, "Content-Type": content_type}
    return {"statusCode": status, "headers": headers, "body": body_s}


def _path(event: dict) -> str:
    raw = event.get("rawPath") or event.get("path") or "/"
    # Function URL may include stage; strip trailing slash except root
    if raw != "/" and raw.endswith("/"):
        raw = raw[:-1]
    return raw


def handler(event, context):
    method = (
        (event.get("requestContext") or {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or "GET"
    ).upper()
    path = _path(event)

    if method == "OPTIONS":
        return _resp(204, "")

    if method == "GET" and path in ("/", "/index.html", ""):
        return _resp(200, HTML, "text/html; charset=utf-8")

    if method == "GET" and path == "/health":
        load_catalog()
        return _resp(200, {
            "ok": True,
            "endpoint": ENDPOINT,
            "model": "stage-a-torch-v4",
            "catalog_n": len(_CATALOG or []),
            "temporary": True,
        })

    if method == "POST" and path in ("/api/rank", "/rank"):
        try:
            body_raw = event.get("body") or "{}"
            if event.get("isBase64Encoded"):
                import base64
                body_raw = base64.b64decode(body_raw).decode("utf-8")
            body = json.loads(body_raw) if isinstance(body_raw, str) else (body_raw or {})
        except Exception as e:
            return _resp(400, {"error": f"invalid JSON: {e}"})

        utterance = (body.get("utterance") or body.get("user_utterance") or "").strip()
        if not utterance:
            return _resp(400, {"error": "utterance required"})
        department = body.get("department") or None
        pin = body.get("pin") or None
        top_k = int(body.get("top_k") or body.get("top_n") or 5)
        pool_limit = int(body.get("pool_limit") or POOL_LIMIT)

        cands = select_candidates(utterance, department, pin, limit=pool_limit)
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
            out, rtt = invoke_endpoint(payload)
        except Exception as e:
            return _resp(502, {"error": f"InvokeEndpoint failed: {e}", "endpoint": ENDPOINT})

        ranked = demote_noop_ranked(list(out.get("ranked") or []), utterance, pin, top_k)
        ranked = boost_ranked_for_intent(ranked, utterance, top_k)
        fams_now = detect_families(utterance, pin)
        actiony = is_action_like(utterance, pin) or bool(fams_now & _STRONG_ACTION_FAMILIES) or has_action_tokens(utterance)
        # Force no_tool hint consistent with gate — action/Hinglish NEVER yellow no_tool
        if wants_no_op(utterance):
            no_tool = True
        elif actiony:
            no_tool = False
            # belt-and-suspenders: strip no_op again if SM stuffed it
            ranked = [r for r in ranked if (r.get("tool") or r.get("name")) != META_NOOP][:top_k]
        else:
            no_tool = bool(out.get("no_tool"))
        if wants_no_op(utterance) and ranked and ranked[0].get("tool") != META_NOOP:
            # ensure no_op tops for explicit chitchat when present in ranked/raw
            noop_row = next((r for r in (out.get("ranked") or []) if r.get("tool") == META_NOOP), None)
            if noop_row:
                ranked = [noop_row] + [r for r in ranked if r.get("tool") != META_NOOP]
                ranked = ranked[:top_k]
        return _resp(200, {
            "utterance": utterance,
            "ranked": ranked,
            "no_tool": no_tool,
            "meta_first": out.get("meta_first"),
            "policy_version": (out.get("policy_version") or "") + "+noop_gate_v2",
            "endpoint": ENDPOINT,
            "model": "stage-a-torch-v4",
            "stage": "endpoint",
            "temporary": True,
            "latency_ms": {"total": round(rtt, 2)},
            "timing": {
                "backend": "sagemaker",
                "rtt_ms": round(rtt, 2),
                "n_candidates": len(cands),
                "note": "SageMaker serverless InvokeEndpoint wall-clock RTT (not model-only). Temporary AWS demo. no_op gated.",
            },
            "candidate_sample": [c["name"] for c in cands[:12]],
            "families": sorted(detect_families(utterance, pin)),
            "noop_gated": True,
        })

    return _resp(404, {"error": "not found", "path": path})
