import os
import io
import json
import time
import math
import hashlib
import threading
from typing import List, Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import httpx

# --------------------------------------------------------------------------------------
# Simple globals (in-memory). No DB to keep it simple.
# --------------------------------------------------------------------------------------
app = FastAPI(title="Media Buying Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

_LAST_ERROR: Optional[str] = None
_DF: Optional[pd.DataFrame] = None  # last ingested raw df
_DF_AGG: Optional[pd.DataFrame] = None  # last aggregated by ad
_LOCK = threading.Lock()

# --------------------------------------------------------------------------------------
# Heuristics & thresholds (Shaun/Spencer-informed)
# --------------------------------------------------------------------------------------
DEFAULTS = {
    # Spend gates
    "min_spend_for_decision": 20.0,     # below this => "iterate / learn more"
    "min_spend_for_scale": 50.0,        # more confidence to scale

    # ROAS/CPR (CPR == CPA on Meta)
    "target_roas": 2.0,                 # change per account
    "breakeven_cpr": 30.0,              # change per account (your "CPR")

    # Creative soft metrics
    "good_hook": 0.30,                  # 30% hook = good; 0.40 great; 0.50 elite
    "great_hook": 0.40,
    "elite_hook": 0.50,
    "good_hold": 0.10,                  # 10% hold = good; 0.15 great
    "great_hold": 0.15,
    "ctr_ok": 1.0,                      # % CTR >= 1% is OK; 1.5% good; 2% great
    "ctr_good": 1.5,
    "ctr_great": 2.0,

    # Funnel drop thresholds (relative falloffs)
    "bad_click_to_atc": 0.90,           # >90% drop from clicks -> ATC suggests LP trust/offer issues
    "bad_atc_to_ic": 0.60,              # >60% drop ATC -> Initiated Checkout suggests friction/trust/fees
    "bad_ic_to_purchase": 0.60,         # >60% drop IC -> Purchase suggests payment/returns/shipping issues
}

# Creative angle mix target (example)
ANGLE_MIX_TARGET = {"pain": 40, "curiosity": 30, "proof": 20, "social": 10}

# --------------------------------------------------------------------------------------
# SerpAPI tiny client (optional). Auto-disabled if env missing.
# --------------------------------------------------------------------------------------
_SERP_CACHE: Dict[str, Any] = {}
def serp_status() -> Dict[str, Any]:
    enabled = os.getenv("SERPAPI_ENABLE", "0") == "1"
    budget = int(os.getenv("SERPAPI_MAX_DAILY_CALLS", "100"))
    day = time.strftime("%Y-%m-%d")
    used = _SERP_CACHE.get(f"calls_{day}", 0)
    return {"enabled": enabled, "daily_budget": budget, "used_today": used, "remaining": max(budget - used, 0)}

def serp_google(q: str, num: int = 8) -> Dict[str, Any]:
    if os.getenv("SERPAPI_ENABLE", "0") != "1":
        return {"disabled": True}
    key = os.getenv("SERPAPI_API_KEY", "")
    if not key:
        return {"disabled": True, "reason": "SERPAPI_API_KEY not set"}
    params = {"engine": "google", "q": q, "num": num, "api_key": key}
    cache_key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    hit = _SERP_CACHE.get(cache_key)
    now = time.time()
    if hit and now - hit["t"] < 86400:
        return hit["data"]
    with httpx.Client(timeout=20.0) as client:
        r = client.get("https://serpapi.com/search.json", params=params)
        r.raise_for_status()
        data = r.json()
    day = time.strftime("%Y-%m-%d")
    _SERP_CACHE[cache_key] = {"t": now, "data": data}
    _SERP_CACHE[f"calls_{day}"] = _SERP_CACHE.get(f"calls_{day}", 0) + 1
    return data

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _set_error(e: Exception):
    global _LAST_ERROR
    _LAST_ERROR = f"{type(e).__name__}: {str(e)}"

def _num(x) -> float:
    try:
        if pd.isna(x) or x == "" or x is None:
            return 0.0
        return float(x)
    except Exception:
        try:
            return float(str(x).replace(",", ""))
        except Exception:
            return 0.0

def normalize_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map common Meta export columns to a canonical schema.
    Handles Website purchases vs purchases, and ROAS-derived revenue if missing.
    """
    df = raw.copy()

    # Canonical names (best effort based on your sample)
    colmap = {
        "Ad name": "ad_name",
        "Day": "date",
        "Amount spent (USD)": "spend",
        "purchases": "purchases",
        "Website purchases": "purchases_web",
        "Purchase conversion value": "revenue",
        "purchase conversion value": "revenue",
        "Purchase ROAS (return on ad spend)": "roas",
        "Website purchase ROAS (return on advertising spend)": "roas_web",
        "Link clicks": "link_clicks",
        "Adds to cart": "atc",
        "Website adds to cart": "atc_web",
        "Checkouts initiated": "ic",
        "Website checkouts initiated": "ic_web",
        "CTR (link click-through rate)": "ctr_pct",
        "hook rate": "hook_rate",
        "hold rate": "hold_rate",
        "Reach": "reach",
        "Reporting starts": "reporting_start",
        "Reporting ends": "reporting_end",
        "Video average play time": "avg_play_time"
    }

    # Copy columns if present
    for k, v in colmap.items():
        if k in df.columns:
            df[v] = df[k]

    # Fallbacks
    if "purchases" not in df.columns and "purchases_web" in df.columns:
        df["purchases"] = df["purchases_web"]
    if "roas" not in df.columns and "roas_web" in df.columns:
        df["roas"] = df["roas_web"]

    # Coerce numerics
    for c in ["spend", "purchases", "revenue", "roas", "link_clicks",
              "atc", "ic", "ctr_pct", "hook_rate", "hold_rate", "reach", "avg_play_time"]:
        if c in df.columns:
            df[c] = df[c].apply(_num)

    # Compute revenue if missing but ROAS & spend exist
    if "revenue" not in df.columns:
        if "roas" in df.columns and "spend" in df.columns:
            df["revenue"] = df["roas"] * df["spend"]
        else:
            df["revenue"] = 0.0

    # Compute CPR (Meta "CPR" == CPA) if we have purchases
    if "spend" in df.columns and "purchases" in df.columns:
        df["cpr"] = df.apply(lambda r: (r["spend"] / r["purchases"]) if r["purchases"] > 0 else 0.0, axis=1)
    else:
        df["cpr"] = 0.0

    # Ensure ad_name exists
    if "ad_name" not in df.columns:
        # try other guesses
        for guess in ["Ad", "Ad title", "Ad ID"]:
            if guess in df.columns:
                df["ad_name"] = df[guess].astype(str)
                break
    if "ad_name" not in df.columns:
        df["ad_name"] = "Unknown"

    # Date normalization
    if "date" in df.columns:
        # Keep as string; parsing not necessary for aggregation
        df["date"] = df["date"].astype(str)

    # Fill NA
    df = df.fillna(0)

    # Keep only columns we use downstream
    keep = ["ad_name", "date", "spend", "purchases", "revenue", "roas", "cpr",
            "link_clicks", "atc", "ic", "ctr_pct", "hook_rate", "hold_rate", "reach"]
    for k in keep:
        if k not in df.columns:
            df[k] = 0.0 if k not in ["ad_name", "date"] else ""

    return df

def aggregate_by_ad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum hard counts; average soft rates.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    hard_sum = df.groupby("ad_name", dropna=False).agg({
        "spend": "sum",
        "purchases": "sum",
        "revenue": "sum",
        "link_clicks": "sum",
        "atc": "sum",
        "ic": "sum",
        "reach": "sum"
    })

    # Soft rates: simple mean (you can change to weighted if you add impressions later)
    soft_mean = df.groupby("ad_name", dropna=False).agg({
        "roas": "mean",
        "cpr": "mean",
        "ctr_pct": "mean",
        "hook_rate": "mean",
        "hold_rate": "mean"
    })

    agg = hard_sum.join(soft_mean, how="left").reset_index()

    # Recompute safer CPR + ROAS from sums as well
    agg["roas_sum"] = agg.apply(lambda r: (r["revenue"] / r["spend"]) if r["spend"] > 0 else 0.0, axis=1)
    agg["cpr_sum"] = agg.apply(lambda r: (r["spend"] / r["purchases"]) if r["purchases"] > 0 else 0.0, axis=1)

    # Use sum-based versions primarily
    agg["roas_final"] = agg["roas_sum"].where(agg["roas_sum"] > 0, agg["roas"])
    agg["cpr_final"] = agg["cpr_sum"].where(agg["cpr_sum"] > 0, agg["cpr"])

    return agg

def decide_actions(agg: pd.DataFrame, cfg: Dict[str, float]) -> Dict[str, Any]:
    scale, kill, iterate, potential = [], [], [], []

    for _, r in agg.iterrows():
        name = str(r["ad_name"])
        spend = float(r["spend"])
        roas = float(r["roas_final"])
        cpr = float(r["cpr_final"])
        hook = float(r["hook_rate"])
        hold = float(r["hold_rate"])
        ctr = float(r["ctr_pct"])
        clicks = float(r["link_clicks"])
        atc = float(r["atc"])
        ic = float(r["ic"])
        pu = float(r["purchases"])

        # Funnel ratios (avoid div by 0)
        drop_clicks_atc = (1 - (atc / clicks)) if clicks > 0 else None
        drop_atc_ic = (1 - (ic / atc)) if atc > 0 else None
        drop_ic_pu = (1 - (pu / ic)) if ic > 0 else None

        # Flags for funnel issues
        funnel_flags = []
        if drop_clicks_atc is not None and drop_clicks_atc > cfg["bad_click_to_atc"]:
            funnel_flags.append("Huge drop Clicks→ATC (LP trust/offer/UX)")
        if drop_atc_ic is not None and drop_atc_ic > cfg["bad_atc_to_ic"]:
            funnel_flags.append("Big drop ATC→Checkout (fees/shipping/friction)")
        if drop_ic_pu is not None and drop_ic_pu > cfg["bad_ic_to_purchase"]:
            funnel_flags.append("Big drop Checkout→Purchase (payments/returns/confidence)")

        # Potential winner logic: strong attention metrics but not enough spend or weak conv
        strong_attention = (hook >= cfg["good_hook"]) or (hold >= cfg["good_hold"]) or (ctr >= cfg["ctr_good"])
        low_spend = spend < cfg["min_spend_for_decision"]
        poor_conv = pu == 0 and (clicks > 20 or ctr >= cfg["ctr_ok"])

        if strong_attention and (low_spend or poor_conv):
            potential.append({
                "ad_id": name,
                "name": name,
                "spend": round(spend, 2),
                "roas": round(roas, 2),
                "cpr": round(cpr, 2),
                "hook": round(hook, 3),
                "hold": round(hold, 3),
                "ctr": round(ctr, 3),
                "reason": "Strong attention (hook/hold/CTR). Needs more spend or LP/offer check.",
                "funnel_flags": funnel_flags
            })

        # Decision gating
        if spend >= cfg["min_spend_for_decision"]:
            if (roas >= cfg["target_roas"]) or (cpr > 0 and cpr <= cfg["breakeven_cpr"]):
                scale.append({
                    "ad_id": name,
                    "name": name,
                    "spend": round(spend, 2),
                    "roas": round(roas, 2),
                    "cpr": round(cpr, 2),
                    "reason": "Meets/exceeds KPI. Consider +20% budget."
                })
            elif (roas == 0 and pu == 0 and spend >= cfg["min_spend_for_scale"]) or (cpr > cfg["breakeven_cpr"] * 1.2):
                kill.append({
                    "ad_id": name,
                    "name": name,
                    "spend": round(spend, 2),
                    "roas": round(roas, 2),
                    "cpr": round(cpr, 2),
                    "reason": "Below breakeven or no conversions with enough spend.",
                    "funnel_flags": funnel_flags
                })
            else:
                iterate.append({
                    "ad_id": name,
                    "name": name,
                    "spend": round(spend, 2),
                    "roas": round(roas, 2),
                    "cpr": round(cpr, 2),
                    "reason": "Between breakeven and target. Let learn or iterate creative/LP.",
                    "funnel_flags": funnel_flags
                })
        else:
            iterate.append({
                "ad_id": name,
                "name": name,
                "spend": round(spend, 2),
                "roas": round(roas, 2),
                "cpr": round(cpr, 2),
                "reason": f"Not enough spend (<{cfg['min_spend_for_decision']}).",
                "funnel_flags": funnel_flags
            })

    # Creative gaps: suggest number of new ads (toy logic: ensure at least 8 in test mix)
    needed_new = max(0, 8 - len(agg.index))
    return {
        "scale": scale,
        "potential_winners": potential,
        "kill": kill,
        "iterate": iterate,
        "creative_gaps": {
            "needed_new_creatives": needed_new,
            "angle_mix": ANGLE_MIX_TARGET,
            "bans": []
        }
    }

def diagnose_account(agg: pd.DataFrame, cfg: Dict[str, float]) -> Dict[str, Any]:
    # Aggregate account-level funnel & soft metrics
    totals = {
        "spend": agg["spend"].sum(),
        "revenue": agg["revenue"].sum(),
        "purchases": agg["purchases"].sum(),
        "clicks": agg["link_clicks"].sum(),
        "atc": agg["atc"].sum(),
        "ic": agg["ic"].sum(),
    }
    # Drop-offs
    def safe_rate(a, b):
        return (a / b) if b > 0 else None

    click_to_atc = safe_rate(totals["atc"], totals["clicks"])
    atc_to_ic = safe_rate(totals["ic"], totals["atc"])
    ic_to_pu = safe_rate(totals["purchases"], totals["ic"])

    notes = []
    if click_to_atc is not None and (1 - click_to_atc) > cfg["bad_click_to_atc"]:
        notes.append("Severe drop Clicks→ATC: test LP trust (social proof, clear price, shipping, returns), faster load, benefit-first headlines.")
    if atc_to_ic is not None and (1 - atc_to_ic) > cfg["bad_atc_to_ic"]:
        notes.append("Big drop ATC→Checkout: unexpected fees/shipping, coupon fields, account creation friction.")
    if ic_to_pu is not None and (1 - ic_to_pu) > cfg["bad_ic_to_purchase"]:
        notes.append("Big drop Checkout→Purchase: payment options, error rates, returns policy clarity, mobile UX.")

    return {
        "account_funnel": {
            "click_to_atc": round(click_to_atc, 3) if click_to_atc is not None else None,
            "atc_to_ic": round(atc_to_ic, 3) if atc_to_ic is not None else None,
            "ic_to_purchase": round(ic_to_pu, 3) if ic_to_pu is not None else None,
        },
        "notes": notes
    }

# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    # Plain string (NOT f-string) to avoid curly-brace issues
    html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Media Buying Agent</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
:root { color-scheme: dark; }
body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; background:#0b0f14; color:#e6edf3; }
header { padding:16px 20px; border-bottom:1px solid #1f2833; position:sticky; top:0; background:#0b0f14; }
h1 { margin:0; font-size:18px; letter-spacing:0.5px; }
.tabs { display:flex; gap:8px; padding:12px 20px; border-bottom:1px solid #1f2833; }
.tab { padding:8px 12px; border:1px solid #1f2833; border-radius:10px; cursor:pointer; background:#121820; }
.tab.active { background:#1b2330; border-color:#2a3646; }
section { display:none; padding:20px; max-width:1100px; }
section.active { display:block; }
.card { background:#0f1520; border:1px solid #1f2833; border-radius:14px; padding:16px; margin:12px 0; }
input, textarea, button { background:#0f1520; color:#e6edf3; border:1px solid #2a3646; border-radius:10px; padding:10px; width:100%; }
button { cursor:pointer; background:#182235; }
pre { white-space:pre-wrap; word-wrap:break-word; }
small { color:#8aa1b5; }
label { display:block; margin:8px 0 6px; color:#c7d5e0; }
.grid { display:grid; gap:12px; grid-template-columns: repeat(2, 1fr); }
@media (max-width:900px){ .grid { grid-template-columns: 1fr; } }
.code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:13px; }
</style>
</head>
<body>
<header><h1>Media Buying Agent</h1></header>
<div class="tabs">
  <div class="tab active" data-t="ingest">Ingest CSV</div>
  <div class="tab" data-t="analyze">Analyze</div>
  <div class="tab" data-t="coach">Coach</div>
  <div class="tab" data-t="prompt">Prompt Lab</div>
  <div class="tab" data-t="script">Script Analyzer</div>
</div>

<section id="ingest" class="active">
  <div class="card">
    <label>Upload Meta CSV</label>
    <input id="csv" type="file" accept=".csv" />
    <div class="grid">
      <button onclick="ingest()">Ingest</button>
      <button onclick="ingestDebug()">Parse Only (Debug)</button>
    </div>
    <pre id="ingest_out" class="code"></pre>
    <small>Tip: Any date range is fine. We aggregate by Ad name.</small>
  </div>
</section>

<section id="analyze">
  <div class="card">
    <div class="grid">
      <div>
        <label>Target ROAS</label>
        <input id="target_roas" type="number" step="0.01" value="2.0"/>
      </div>
      <div>
        <label>Breakeven CPR (Meta CPR = CPA)</label>
        <input id="breakeven_cpr" type="number" step="0.01" value="30"/>
      </div>
      <div>
        <label>Min Spend for Decision</label>
        <input id="min_spend" type="number" step="0.01" value="20"/>
      </div>
      <div>
        <label>Min Spend for Kill Check</label>
        <input id="min_spend_scale" type="number" step="0.01" value="50"/>
      </div>
    </div>
    <button onclick="runAnalyze()">Run Analysis</button>
    <pre id="analyze_out" class="code"></pre>
  </div>
</section>

<section id="coach">
  <div class="card">
    <label>Question for the Coach</label>
    <textarea id="coach_q" rows="5" placeholder="e.g., Why is my drop-off from link click to ATC so high?"></textarea>
    <div class="grid">
      <div><label><input id="use_serp" type="checkbox" /> Enrich with outside market context</label></div>
      <div><label>Niche (optional)<input id="coach_niche" /></label></div>
    </div>
    <button onclick="askCoach()">Ask</button>
    <pre id="coach_out" class="code"></pre>
  </div>
</section>

<section id="prompt">
  <div class="card">
    <label>What do you need a prompt for?</label>
    <textarea id="prompt_need" rows="4" placeholder="e.g., UGC storyboard for 30s, audience: men 45+, pain: knee pain, product: brace"></textarea>
    <button onclick="genPrompt()">Generate High-Value Prompt</button>
    <pre id="prompt_out" class="code"></pre>
  </div>
</section>

<section id="script">
  <div class="card">
    <label>Paste your script</label>
    <textarea id="script_text" rows="10" placeholder="Paste script here..."></textarea>
    <button onclick="analyzeScript()">Analyze Script</button>
    <pre id="script_out" class="code"></pre>
  </div>
</section>

<script>
const tabs = document.querySelectorAll('.tab');
tabs.forEach(t => t.onclick = () => {
  tabs.forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
  document.getElementById(t.dataset.t).classList.add('active');
});

async function ingest() {
  const f = document.getElementById('csv').files[0];
  if (!f) { alert('Choose a CSV first'); return; }
  const fd = new FormData();
  fd.append('file', f);
  const r = await fetch('/ingest_csv', { method:'POST', body: fd });
  const j = await r.json();
  document.getElementById('ingest_out').textContent = JSON.stringify(j, null, 2);
}

async function ingestDebug() {
  const f = document.getElementById('csv').files[0];
  if (!f) { alert('Choose a CSV first'); return; }
  const fd = new FormData();
  fd.append('file', f);
  const r = await fetch('/ingest_csv_debug', { method:'POST', body: fd });
  const j = await r.json();
  document.getElementById('ingest_out').textContent = JSON.stringify(j, null, 2);
}

async function runAnalyze() {
  const payload = {
    target_roas: parseFloat(document.getElementById('target_roas').value),
    breakeven_cpr: parseFloat(document.getElementById('breakeven_cpr').value),
    min_spend_for_decision: parseFloat(document.getElementById('min_spend').value),
    min_spend_for_scale: parseFloat(document.getElementById('min_spend_scale').value)
  };
  const r = await fetch('/analyze', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
  const j = await r.json();
  document.getElementById('analyze_out').textContent = JSON.stringify(j, null, 2);
}

async function askCoach() {
  const payload = {
    question: document.getElementById('coach_q').value || "",
    use_serp: document.getElementById('use_serp').checked,
    niche: document.getElementById('coach_niche').value || ""
  };
  const r = await fetch('/coach', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
  const j = await r.json();
  document.getElementById('coach_out').textContent = JSON.stringify(j, null, 2);
}

async function genPrompt() {
  const payload = { need: document.getElementById('prompt_need').value || "" };
  const r = await fetch('/prompt_lab', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
  const j = await r.json();
  document.getElementById('prompt_out').textContent = JSON.stringify(j, null, 2);
}

async function analyzeScript() {
  const payload = { script: document.getElementById('script_text').value || "" };
  const r = await fetch('/script_analyze', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
  const j = await r.json();
  document.getElementById('script_out').textContent = JSON.stringify(j, null, 2);
}
</script>
</body>
</html>
"""
    return HTMLResponse(html)

@app.get("/debug/last_error")
def last_error():
    return {"last_error": _LAST_ERROR}

# --------------------------------------------------------------------------------------
# CSV ingest
# --------------------------------------------------------------------------------------
@app.post("/ingest_csv_debug")
async def ingest_csv_debug(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        return {
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "sample": df.fillna("").head(5).to_dict(orient="records")
        }
    except Exception as e:
        _set_error(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest_csv")
async def ingest_csv(file: UploadFile = File(...)):
    global _DF, _DF_AGG
    try:
        content = await file.read()
        raw = pd.read_csv(io.BytesIO(content))
        df = normalize_df(raw)
        agg = aggregate_by_ad(df)
        with _LOCK:
            _DF = df
            _DF_AGG = agg
        return {
            "ok": True,
            "rows": int(len(df)),
            "ads": int(len(agg)),
            "columns": list(df.columns)[:20]
        }
    except Exception as e:
        _set_error(e)
        return JSONResponse(status_code=500, content={"detail": "Server failed to ingest CSV. Hit /debug/last_error for details."})

# --------------------------------------------------------------------------------------
# Analyze (decisions + funnel diagnosis)
# --------------------------------------------------------------------------------------
@app.post("/analyze")
def analyze(payload: Dict[str, Any] = Body(default={})):
    global _DF_AGG
    try:
        if _DF_AGG is None or _DF_AGG.empty:
            raise HTTPException(status_code=400, detail="No data ingested yet. Upload a CSV first.")

        cfg = DEFAULTS.copy()
        for k in ["target_roas", "breakeven_cpr", "min_spend_for_decision", "min_spend_for_scale"]:
            if k in payload and isinstance(payload[k], (int, float)):
                cfg[k] = float(payload[k])

        actions = decide_actions(_DF_AGG, cfg)
        acct = diagnose_account(_DF_AGG, cfg)

        return actions | {"account_summary": acct}
    except HTTPException:
        raise
    except Exception as e:
        _set_error(e)
        raise HTTPException(status_code=500, detail="Analyze failed. See /debug/last_error")

# --------------------------------------------------------------------------------------
# Coach (reasoning + optional outside context via SerpAPI)
# --------------------------------------------------------------------------------------
@app.post("/coach")
def coach(payload: Dict[str, Any] = Body(default={})):
    try:
        q = (payload.get("question") or "").strip()
        use_serp = bool(payload.get("use_serp", False))
        niche = (payload.get("niche") or "").strip()

        if not q:
            raise HTTPException(status_code=400, detail="Ask a question.")

        # Pull basic account context if available
        agg_preview = None
        if _DF_AGG is not None and not _DF_AGG.empty:
            top = _DF_AGG.sort_values("spend", ascending=False).head(10)
            agg_preview = top[["ad_name","spend","roas_final","cpr_final","hook_rate","hold_rate","ctr_pct","link_clicks","atc","ic","purchases"]].fillna(0).round(3).to_dict(orient="records")

        # Heuristic diagnosis snippet
        diagnosis = []
        if "drop" in q.lower() or "drop-off" in q.lower() or "dropoff" in q.lower():
            # Use account-level diagnosis if available
            cfg = DEFAULTS.copy()
            acct = diagnose_account(_DF_AGG, cfg) if (_DF_AGG is not None and not _DF_AGG.empty) else {}
            diagnosis.append({
                "account_funnel": acct.get("account_funnel"),
                "notes": acct.get("notes", []),
                "quick_tests": [
                    "LP: add trust blocks near price (reviews, UGC quotes, badges).",
                    "Reduce first paint & LCP (<2.5s). Remove heavy scripts.",
                    "Checkout: collapse coupon, show shipping/returns early, add Shop Pay/PayPal/Apple Pay.",
                ]
            })
        else:
            diagnosis.append({
                "general": [
                    "Prioritize ads with spend+results. Soft metrics (CPC/CTR) are directional only.",
                    "If 72h above KPI and >60% click-attributed, scale +20% daily.",
                    "If below breakeven and not at minimum daily spend, reduce 20%; otherwise launch new DCTs and fix funnel."
                ]
            })

        external = None
        if use_serp:
            status = serp_status()
            if status.get("enabled"):
                query = (niche + " " + q).strip() if niche else q
                res = serp_google(query + " marketing best practices tips causes")
                snippets = []
                try:
                    org = res.get("organic_results") or []
                    for r in org[:6]:
                        s = r.get("snippet") or r.get("title") or ""
                        if s: snippets.append(s)
                except Exception:
                    pass
                external = {
                    "serp_status": status,
                    "snippets": snippets[:5]
                }
            else:
                external = {"serp_status": status, "note": "Set SERPAPI_ENABLE=1 and SERPAPI_API_KEY to enable."}

        return {
            "question": q,
            "niche": niche or None,
            "data_preview": agg_preview,
            "diagnosis": diagnosis,
            "external_context": external
        }
    except HTTPException:
        raise
    except Exception as e:
        _set_error(e)
        raise HTTPException(status_code=500, detail="Coach failed. See /debug/last_error")

# --------------------------------------------------------------------------------------
# Prompt Lab
# --------------------------------------------------------------------------------------
@app.post("/prompt_lab")
def prompt_lab(payload: Dict[str, Any] = Body(default={})):
    try:
        need = (payload.get("need") or "").strip()
        if not need:
            raise HTTPException(status_code=400, detail="Tell me what you need a prompt for.")
        # Output a few ready-to-use prompt templates
        return {
            "high_value_prompts": [
                f"Act as a senior DTC creative strategist. Create 5 ad angles for: {need}. For each: a) angle summary, b) audience pain/desire, c) 30s UGC storyboard (Hook/Problem/Agitate/Solution/Proof/CTA), d) 3 headline options, e) 2 thumbnail ideas.",
                f"You are a CRO specialist. Audit the landing page for {need}. Identify: trust gaps, clarity issues, friction, and propose 10 quick A/B tests prioritized by ICE score.",
                f"Pretend you are Meta's delivery system. Explain likely reasons an ad stopped spending for {need}, and list steps to revive delivery without resetting learning."
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        _set_error(e)
        raise HTTPException(status_code=500, detail="Prompt lab failed. See /debug/last_error")

# --------------------------------------------------------------------------------------
# Script Analyzer (Hook/Problem/etc) + tips
# --------------------------------------------------------------------------------------
@app.post("/script_analyze")
def script_analyze(payload: Dict[str, Any] = Body(default={})):
    try:
        text = (payload.get("script") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Paste a script.")

        # Very light segmentation heuristics
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        first = " ".join(lines[:3]).lower()
        has_hook = any(x in first for x in ["stop", "wait", "imagine", "did you know", "if you", "tired of"])
        parts = {
            "hook": lines[:2] if has_hook else lines[:1],
            "problem": [],
            "agitation": [],
            "solution": [],
            "proof": [],
            "offer": [],
            "cta": []
        }

        # Naive keyword-based sectioning (you can paste any script; this just gets you a labeled map)
        for l in lines[1:]:
            low = l.lower()
            if any(k in low for k in ["problem", "struggle", "stuck", "pain", "frustrat"]):
                parts["problem"].append(l)
            elif any(k in low for k in ["worse", "costly", "risk", "imagine if you don't", "what happens when"]):
                parts["agitation"].append(l)
            elif any(k in low for k in ["introducing", "here's how", "solution", "we built", "this is why"]):
                parts["solution"].append(l)
            elif any(k in low for k in ["proof", "testimonial", "thousands", "stars", "backed by", "clinical", "case study"]):
                parts["proof"].append(l)
            elif any(k in low for k in ["today only", "limited", "save", "free", "bundle", "discount", "offer"]):
                parts["offer"].append(l)
            elif any(k in low for k in ["tap", "click", "shop", "buy", "get yours", "learn more", "try now"]):
                parts["cta"].append(l)
            else:
                # Put unmatched solution-ish lines under solution by default
                parts["solution"].append(l)

        # Tips based on Shaun/Spencer-style heuristics
        tips = []
        # Hook/hold targets
        tips.append("Aim Hook ≥ 30–40% (3s plays ÷ impressions) and Hold ≥ 10–15% (throughplays ÷ impressions).")
        tips.append("Open with an undeniable pattern interrupt: motion change, bold claim with specificity, or visual curiosity.")
        tips.append("Speak to one pain/desire clearly; don't stack 4 claims in the first 5 seconds.")
        tips.append("Insert proof early: star rating pop-in, quick testimonial lower third, or before/after visual by 5–7s.")
        tips.append("CTA: repeat twice—mid and end. Keep it action + outcome oriented (e.g., “Tap to relieve knee pain fast”).")

        return {"sections": parts, "suggestions": tips}
    except HTTPException:
        raise
    except Exception as e:
        _set_error(e)
        raise HTTPException(status_code=500, detail="Script analyze failed. See /debug/last_error")
