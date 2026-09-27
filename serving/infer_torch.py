#!/usr/bin/env python3
"""Load Stage A torch checkpoint; score candidates; emit TRAINING_IO output JSON."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn as nn

TOKEN_RE = re.compile(r"[a-z0-9_./@+-]+", re.I)
POLICY = "jev-stage-a-v3-clean+fam_conflict_v1"
DEFAULT_TAU = 0.25


class HashBag(nn.Module):
    def __init__(self, n_buckets: int, emb_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.n_buckets = n_buckets
        self.emb = nn.EmbeddingBag(n_buckets, emb_dim, mode="mean", sparse=False)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, bag: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        x = self.emb(bag, offsets)
        return self.mlp(x).squeeze(-1)


def tokenize(text: str) -> list[str]:
    toks = TOKEN_RE.findall((text or "").lower())
    out = list(toks)
    for a, b in zip(toks, toks[1:]):
        out.append(f"{a}#{b}")
    return out


def hash_token(tok: str, n_buckets: int) -> int:
    h = 2166136261
    for ch in tok.encode("utf-8", errors="ignore"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h % n_buckets


def feature_text(state: dict, cand: dict) -> str:
    utt = state.get("user_utterance") or state.get("intent_text") or ""
    dept = state.get("department") or "null"
    pin = state.get("pin") or "null"
    prior = state.get("prior_tools") or []
    prior_s = " ".join(prior) if isinstance(prior, list) else str(prior)
    channel = state.get("channel") or "null"
    name = cand.get("name") or cand.get("federated_name") or ""
    cdept = cand.get("department") or dept
    risk = cand.get("risk_level") or "low"
    desc = (cand.get("description") or "")[:160]
    return (
        f"{utt} | dept:{dept} | pin:{pin} | prior:{prior_s} | ch:{channel} | "
        f"tool:{name} | d:{cdept} | risk:{risk} | desc:{desc}"
    )


def load_model(model_dir: Path) -> tuple[HashBag, dict]:
    cfg = json.loads((model_dir / "config.json").read_text())
    ckpt = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    n_buckets = ckpt.get("n_buckets", cfg["n_buckets"])
    emb_dim = ckpt.get("emb_dim", cfg["emb_dim"])
    hidden = ckpt.get("hidden", cfg["hidden"])
    model = HashBag(n_buckets, emb_dim, hidden)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, {**cfg, "n_buckets": n_buckets}


@torch.no_grad()
def score_texts(model: HashBag, texts: list[str], n_buckets: int) -> list[float]:
    flat, offsets, off = [], [], 0
    for text in texts:
        ids = [hash_token(t, n_buckets) for t in tokenize(text)] or [0]
        offsets.append(off)
        flat.extend(ids)
        off += len(ids)
    bag = torch.tensor(flat, dtype=torch.long)
    offs = torch.tensor(offsets, dtype=torch.long)
    logits = model(bag, offs)
    return torch.sigmoid(logits).tolist()


def filter_candidates(state: dict, candidates: list[dict]) -> list[dict]:
    pin = state.get("pin")
    avail = set(state.get("available_tools") or [])
    out = []
    for c in candidates:
        name = c.get("name") or c.get("federated_name")
        if not name:
            continue
        if avail and name not in avail:
            continue
        if pin:
            # pin filter: federated ext.{slug_underscored}.* or server_slug match
            slug_u = pin.replace("-", "_")
            ok = (
                name.startswith(f"ext.{slug_u}.")
                or c.get("server_slug") == pin
                or (c.get("department") and False)
            )
            # also allow native tools that mention pin loosely? TRAINING_IO: filter to that slug
            if not ok:
                continue
        out.append({**c, "name": name})
    return out if out else [{**c, "name": c.get("name") or c.get("federated_name")} for c in candidates if c.get("name") or c.get("federated_name")]


def needs_approval(cand: dict) -> bool:
    risk = (cand.get("risk_level") or "").lower()
    return risk in ("high", "critical")


def is_meta(name: str) -> bool:
    return name in ("discover_capabilities", "workspace_tools_snapshot", "oauth_status", "tool_schema_get")


# --- family conflict policy (social vs mail) ---
_EMAIL_WORDS = ("email", "e-mail", "gmail", "mail", "inbox", "compose", "proposal", "outlook")
_SOCIAL_POST_WORDS = (
    "reel", "reels", "story", "stories", "post", "publish", "schedule post",
    "instagram", "threads", "facebook", "twitter", "x.com", "reddit", "buffer", "ayrshare", "social",
)
_SOCIAL_DM_WORDS = ("dm", "dms", "direct message", "whatsapp", "slack dm")
_MAIL_NAME = ("mail", "gmail", "email", "outlook", "smtp")
_SOCIAL_POST_NAME = ("ayrshare", "buffer", "instagram", "ig_graph_media", "threads_create", "facebook_create", "twitter", "publish", "schedule_post")
_SOCIAL_DM_NAME = ("instagram_send_dm", "instagram_list_dm", "instagram_get_conversation", "instagram_get_message", "instagram_list_pending", "ig_graph_conversation", "whatsapp", "post_message")


_DM_WORD_RE = re.compile(r"(?<![a-z0-9])(dms?|direct[\s-]?messages?)(?![a-z0-9])", re.I)


def _detect_utt_families(utt: str) -> set[str]:
    u = (utt or "").lower()
    fams: set[str] = set()
    if any(w in u for w in _EMAIL_WORDS):
        fams.add("mail_send")
    if any(w in u for w in _SOCIAL_POST_WORDS) or any(x in u for x in ("reel", "reels", "story", "stories")):
        fams.add("social_post")
    if _DM_WORD_RE.search(u) or any(w in u for w in ("direct message", "whatsapp", "slack dm")):
        if "mail" not in u and "email" not in u and "gmail" not in u:
            fams.add("social_dm")
    return fams


def _tool_fams(name: str, desc: str = "") -> set[str]:
    blob = f"{name} {desc}".lower()
    tags: set[str] = set()
    if any(x in blob for x in _MAIL_NAME):
        tags.add("mail_send")
    if any(x in blob for x in _SOCIAL_POST_NAME):
        tags.add("social_post")
    if any(x in blob for x in _SOCIAL_DM_NAME):
        tags.add("social_dm")
    return tags


def apply_family_conflict_scores(filtered: list[dict], scores: list[float], utterance: str) -> list[float]:
    fams = _detect_utt_families(utterance)
    social = fams & {"social_post", "social_dm"}
    if not social or "mail_send" in fams:
        return scores
    out = []
    for c, s in zip(filtered, scores):
        name = c.get("name") or ""
        tags = _tool_fams(name, c.get("description") or "")
        sc = float(s)
        if "mail_send" in tags and not (tags & social):
            sc *= 0.15
        elif tags & social:
            sc = min(1.0, sc * 1.35 + 0.05)
        out.append(sc)
    return out



def rank(payload: dict, model: HashBag, n_buckets: int, tau: float = DEFAULT_TAU) -> dict:
    state = payload.get("state") or {}
    candidates = payload.get("candidates") or []
    top_k = int(payload.get("top_k") or 3)
    filtered = filter_candidates(state, candidates)
    texts = [feature_text(state, c) for c in filtered]
    scores = score_texts(model, texts, n_buckets) if texts else []
    scores = apply_family_conflict_scores(filtered, scores, state.get("user_utterance") or "")
    ranked_raw = []
    for c, s in sorted(zip(filtered, scores), key=lambda x: -float(x[1])):
        name = c["name"]
        ranked_raw.append(
            {
                "tool": name,
                "score": float(s),
                "args_sketch": {},
                "meta": is_meta(name),
                "needs_approval": needs_approval(c),
            }
        )
    ranked_raw.sort(key=lambda x: x["score"], reverse=True)
    ranked = ranked_raw[:top_k]
    max_score = ranked[0]["score"] if ranked else 0.0
    no_tool_utt = (state.get("user_utterance") or "").lower().strip()
    no_tool_phrases = (
        "thanks", "thank you", "who are you", "what is jev", "good morning",
        "never mind", "ok cool", "ok.", "okay", "hi", "hello",
    )
    no_tool_hint = any(x in no_tool_utt for x in no_tool_phrases) or no_tool_utt in {
        "thanks!", "thanks", "thank you", "ok cool", "never mind", "good morning",
    }
    # Prefer no_op if present among candidates for chitchat
    if no_tool_hint:
        noop = next((r for r in ranked_raw if r["tool"] in ("no_op", "help")), None)
        if noop:
            ranked = [noop] + [r for r in ranked_raw if r["tool"] != noop["tool"]]
            ranked = ranked[:top_k]
            max_score = ranked[0]["score"] if ranked else 0.0
    meta_first = (not no_tool_hint) and bool(ranked) and max_score < tau
    if meta_first:
        disc = next((r for r in ranked_raw if r["tool"] == "discover_capabilities"), None)
        if disc:
            ranked = [disc] + [r for r in ranked if r["tool"] != "discover_capabilities"]
            ranked = ranked[:top_k]
    return {
        "ranked": ranked if not no_tool_hint else ranked,
        "no_tool": bool(no_tool_hint),
        "meta_first": meta_first,
        "deterministic": True,
        "policy_version": POLICY,
    }


def load_catalog(path: Path) -> dict[str, dict]:
    by = {}
    if not path.exists():
        return by
    with path.open() as f:
        for line in f:
            o = json.loads(line)
            fed = o.get("federated_name") or o.get("name")
            by[fed] = o
    return by


def enrich_candidates(cands: list[dict], catalog: dict[str, dict]) -> list[dict]:
    out = []
    for c in cands:
        name = c.get("name") or c.get("federated_name")
        meta = catalog.get(name) or {}
        out.append(
            {
                **c,
                "name": name,
                "description": c.get("description") or meta.get("description") or "",
                "department": c.get("department") or meta.get("department"),
                "risk_level": c.get("risk_level") or meta.get("risk_level") or "low",
                "server_slug": c.get("server_slug") or meta.get("server_slug"),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("manifests/stage-a-torch-v1"))
    ap.add_argument("--payload", type=Path, help="single JSON payload path")
    ap.add_argument("--payloads", type=Path, help="JSONL or JSON list of payloads")
    ap.add_argument("--catalog", type=Path, default=Path("data/tool_catalog_enriched.jsonl"))
    ap.add_argument("--out", type=Path, help="write results JSON")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    model, cfg = load_model(args.model_dir)
    catalog = load_catalog(args.catalog)
    n_buckets = cfg["n_buckets"]

    payloads = []
    if args.payload:
        payloads.append(json.loads(args.payload.read_text()))
    if args.payloads:
        text = args.payloads.read_text().strip()
        if text.startswith("["):
            payloads.extend(json.loads(text))
        else:
            for line in text.splitlines():
                if line.strip():
                    payloads.append(json.loads(line))

    if not payloads:
        raise SystemExit("provide --payload or --payloads")

    results = []
    for p in payloads:
        p = dict(p)
        p["top_k"] = p.get("top_k", args.top_k)
        p["candidates"] = enrich_candidates(p.get("candidates") or [], catalog)
        out = rank(p, model, n_buckets, tau=args.tau)
        results.append({"input": {"utterance": (p.get("state") or {}).get("user_utterance"), "n_cand": len(p["candidates"])}, "output": out})

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results if len(results) > 1 else results[0], indent=2))
        print(f"wrote {args.out}")
    else:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))


if __name__ == "__main__":
    main()
