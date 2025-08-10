# main.py
# Erto Agent – Media Buying Coach / Analyzer
# - Deeper tips & reasons (Shaun/Spencer playbook baked in)
# - Optional external snippets via SerpAPI used ONLY for tips/suggestions reasoning
# - Keeps the same single-page UI (tabs) you liked

import io
import os
import json
import math
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse

app = FastAPI(title="Erto Agent")

# --------------------------- Globals ---------------------------

LATEST_DF: Optional[pd.DataFrame] = None
LAST_ERROR: Optional[str] = None

DEFAULT_SETTINGS = {
    "breakeven_roas": 1.5,     # tweakable
    "target_roas": 2.0,        # scale threshold
    "min_test_spend": 20.0,    # spend to judge a creative
    "min_scale_spend": 50.0,   # spend to consider scale
    "ctr_good": 0.015,         # 1.5%
    "hook_good": 0.30,         # 30%
    "hold_good": 0.10,         # 10%
    "assume_click_attrib_ok": False,  # if True, skip the 7d-click check gate
}

# ---------------------- Environment toggles --------------------

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in {"1", "true", "yes", "on"}

EXTERNAL_DATA = env_bool("EXTERNAL_DATA", False)
SERP_API_KEY = os.getenv("SERP_API_KEY", "").strip()

# --------------------------- Utils -----------------------------

def to_float(x) -> float:
    if pd.isna(x):
        return 0.0
    try:
        s = str(x).strip().replace(",", "")
        if s == "":
            return 0.0
        return float(s)
    except Exception:
        return 0.0

def pct_to_ratio(v: float) -> float:
    """Handle metrics that might arrive as 0.034 or 3.4 (percent)."""
    if pd.isna(v):
        return 0.0
    v = float(v)
    # If it looks like a percent (e.g., 3.4 for 3.4%), convert to 0.034
    return v / 100.0 if v > 1.0 else v

def ratio_to_pct(v: float) -> float:
    return round(100.0 * float(v), 1)

def pick_first(df: pd.Series, names: List[str], default=0.0, as_str=False):
    for n in names:
        if n in df and not pd.isna(df[n]):
            return str(df[n]) if as_str else to_float(df[n])
    return default if not as_str else ""

def has_any_col(df: pd.DataFrame, cols: List[str]) -> bool:
    return any(c in df.columns for c in cols)

def safe_div(a: float, b: float) -> float:
    return (a / b) if (b and b != 0) else 0.0

# ----------------- External data for suggestions ----------------

async def serp_snippets(queries: List[str], max_per_query: int = 3) -> List[Dict[str, str]]:
    """Fetch short snippets from SerpAPI (Google). Used ONLY for tips/suggestions."""
    if not EXTERNAL_DATA or not SERP_API_KEY:
        return []

    out = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for q in queries:
            try:
                resp = await client.get(
                    "https://serpapi.com/search.json",
                    params={
                        "engine": "google",
                        "q": q,
                        "num": max_per_query,
                        "api_key": SERP_API_KEY,
                    },
                )
                data = resp.json()
                for r in data.get("organic_results", [])[:max_per_query]:
                    title = r.get("title", "")
                    snippet = r.get("snippet", "") or r.get("snippet_highlighted_words", [""])[0]
                    if title or snippet:
                        out.append({"q": q, "title": title[:140], "snippet": snippet[:220]})
            except Exception:
                # Fail silently; external data is optional
                continue
    return out

# -------------------------- Parsing -----------------------------

COLUMN_MAP = {
    "ad_name": ["Ad name", "Ad", "Ad Name", "name"],
    "adset_name": ["Ad set name", "Ad set", "Adset", "Ad set name.1"],
    "campaign_name": ["Campaign name", "Campaign", "Campaign Name"],
    "day": ["Day", "Date", "Reporting starts"],

    "spend": ["Amount spent (USD)", "Spend", "Amount Spent"],
    "purchases": ["Website purchases", "Purchases", "purchases"],
    "purchase_roas": [
        "Website purchase ROAS (return on advertising spend)",
        "Purchase ROAS (return on ad spend)",
        "ROAS",
    ],
    "cost_per_purchase": ["Cost per purchase", "CPP", "Cost per result"],

    "clicks": ["Link clicks", "Unique link clicks", "Outbound clicks", "Unique outbound clicks"],
    "ctr": ["CTR (link click-through rate)", "CTR", "Unique outbound CTR", "Link CTR"],
    "adds_to_cart": ["Website adds to cart", "Adds to cart"],
    "checkouts": ["Website checkouts initiated", "Checkouts initiated"],

    "hook": ["hook rate", "Video hook", "Hook"],
    "hold": ["hold rate", "Video hold", "Hold"],
}

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all referenced columns exist
    for _, cols in COLUMN_MAP.items():
        for c in cols:
            if c in df.columns:
                break
        else:
            # add a missing placeholder filled with zeros
            df[cols[0]] = np.nan

    # Build normalized frame
    out = pd.DataFrame()
    for key, cols in COLUMN_MAP.items():
        for c in cols:
            if c in df.columns:
                out[key] = df[c]
                break
        if key not in out.columns:
            out[key] = np.nan

    # Coerce numeric
    for n in [
        "spend", "purchases", "purchase_roas", "cost_per_purchase",
        "clicks", "ctr", "adds_to_cart", "checkouts", "hook", "hold"
    ]:
        out[n] = out[n].map(to_float)

    # CTR may come as percent; convert to ratio
    out["ctr"] = out["ctr"].map(pct_to_ratio)

    # Hook/Hold may already be ratios (0-1). If >1, treat as percent.
    for n in ["hook", "hold"]:
        out[n] = out[n].map(lambda v: v / 100.0 if v > 1.0 else v)

    # CPR (Meta "cost per result") ≈ CPA in your language.
    # If cost_per_purchase missing, compute from spend/purchases
    out["cpr"] = out.apply(
        lambda r: to_float(r["cost_per_purchase"]) if to_float(r["cost_per_purchase"]) > 0
        else (safe_div(to_float(r["spend"]), to_float(r["purchases"])) if to_float(r["purchases"]) > 0 else 0.0),
        axis=1
    )

    # Final strings
    out["name"] = out["ad_name"].fillna("").astype(str).replace("nan", "", regex=False)
    return out

# --------------------------- Scoring ----------------------------

def load_settings() -> Dict[str, Any]:
    s = DEFAULT_SETTINGS.copy()
    # Allow simple overrides via env (optional)
    for k in list(s.keys()):
        env_name = k.upper()
        if env_name in os.environ:
            try:
                v = os.getenv(env_name)
                if isinstance(s[k], bool):
                    s[k] = env_bool(env_name, s[k])
                else:
                    s[k] = float(v)
            except Exception:
                pass
    return s

def ad_score(row: pd.Series, s: Dict[str, Any]) -> float:
    # Weighted blend: ROAS (capped), spend weight, soft metrics
    roas = to_float(row["purchase_roas"])
    spend = to_float(row["spend"])
    hook = row["hook"]; hold = row["hold"]; ctr = row["ctr"]

    roas_part = min(roas / max(s["target_roas"], 0.01), 2.0)      # up to 2x
    spend_part = min(spend / max(s["min_scale_spend"], 1.0), 1.0) # normalize by scale spend
    soft_part = (
        (1.0 if hook >= s["hook_good"] else 0.0) +
        (1.0 if hold >= s["hold_good"] else 0.0) +
        (1.0 if ctr  >= s["ctr_good"]  else 0.0)
    ) / 3.0
    return round(0.55 * roas_part + 0.25 * spend_part + 0.20 * soft_part, 3)

def creative_reason(row: pd.Series, s: Dict[str, Any]) -> List[str]:
    """Deeper internal reasoning (no external data)."""
    notes = []
    roas = to_float(row["purchase_roas"])
    spend = to_float(row["spend"])
    cpr = to_float(row["cpr"]); purchases = to_float(row["purchases"])
    hook = row["hook"]; hold = row["hold"]; ctr = row["ctr"]

    if spend < s["min_test_spend"]:
        notes.append(f"Not enough spend to judge (< ${s['min_test_spend']:.0f}). Let it run or relaunch in DCT.")
    else:
        if roas >= s["target_roas"]:
            notes.append(f"≥ target ROAS ({roas:.2f}). Watch stability over 72h and confirm ≥60% 7-day click before scaling.")
        elif roas < s["breakeven_roas"] or purchases == 0:
            notes.append("Below breakeven at test spend. Iterate or pause; launch a variant with new hook/angle.")
        else:
            notes.append("Between breakeven and target — treat as ‘promising’ and iterate 1–2 variants.")

    # Soft signal diagnostics (Shaun/Spencer style)
    if hook < s["hook_good"] and hold >= s["hold_good"]:
        notes.append("Hook weak (<30%) but Hold decent — the opening isn’t stopping scroll. Test stronger first line + on-screen text.")
    if hold < s["hold_good"] and hook >= s["hook_good"]:
        notes.append("Hold weak (<10%) — story loses steam after the first beat. Reorder benefits or add social proof by 6–8s.")
    if ctr < s["ctr_good"]:
        notes.append("CTR below 1.5% — creative/story or thumb-stop isn’t resonating with this audience.")
    if cpr == 0 and purchases == 0 and spend >= s["min_test_spend"]:
        notes.append("No purchases at meaningful spend — double-check product-page message match and price framing.")
    return notes

def classify_action(row: pd.Series, s: Dict[str, Any]) -> str:
    roas = to_float(row["purchase_roas"])
    spend = to_float(row["spend"])
    purchases = to_float(row["purchases"])

    if spend < s["min_test_spend"]:
        return "iterate"
    if roas >= s["target_roas"] and spend >= s["min_scale_spend"]:
        return "scale"
    if (roas < s["breakeven_roas"]) or (purchases == 0 and spend >= s["min_test_spend"]):
        return "kill"
    return "iterate"

# --------------- Account-level funnel diagnostics ----------------

def funnel_summary(df: pd.DataFrame) -> Dict[str, Any]:
    clicks = to_float(df["clicks"].sum())
    atc = to_float(df["adds_to_cart"].sum())
    ic = to_float(df["checkouts"].sum())
    purchases = to_float(df["purchases"].sum())

    click_to_atc = safe_div(atc, clicks)
    atc_to_ic = safe_div(ic, atc)
    ic_to_purchase = safe_div(purchases, ic)

    diagnosis = []
    if ic_to_purchase < 0.45:  # aggressive target for DTC
        diagnosis.append("Low Checkout→Purchase (checkout UX friction, payment issues, reassurance missing).")
    if click_to_atc < 0.20:
        diagnosis.append("Low Click→ATC (weak product-page message match, price anchoring, or PDP clarity).")
    if purchases == 0 and clicks > 200:
        diagnosis.append("Traffic but no conversions — verify pixel events + attribution window + PDP load speed.")

    return {
        "blended": {
            "clicks": int(round(clicks)),
            "add_to_cart": int(round(atc)),
            "initiate_checkout": int(round(ic)),
            "purchases": int(round(purchases)),
            "click_to_atc": round(click_to_atc, 4),
            "atc_to_ic": round(atc_to_ic, 4),
            "ic_to_purchase": round(ic_to_purchase, 4),
        },
        "diagnosis": diagnosis or ["Funnel looks healthy at high level."],
    }

# ------------------------ Analyzer core -------------------------

def analyze_core(df_raw: pd.DataFrame) -> Dict[str, Any]:
    s = load_settings()
    df = normalize_df(df_raw.copy())

    # Per-ad frame
    df["score"] = df.apply(lambda r: ad_score(r, s), axis=1)
    df["action"] = df.apply(lambda r: classify_action(r, s), axis=1)
    df["notes"] = df.apply(lambda r: creative_reason(r, s), axis=1)

    # Account summary
    spend = to_float(df["spend"].sum())
    purchases = to_float(df["purchases"].sum())
    roas_vals = df["purchase_roas"].replace(0, np.nan).dropna()
    avg_roas = float(roas_vals.mean()) if not roas_vals.empty else 0.0

    funnel = funnel_summary(df)

    # Potential winners (good score + non-trivial spend)
    winners = (
        df[(df["score"] >= 1.2) & (df["spend"] >= s["min_test_spend"])]
        .sort_values(["score", "spend", "purchase_roas"], ascending=[False, False, False])
    )
    winner_names = winners["name"].fillna("").tolist()[:3]

    # Action buckets
    bucket = {
        "scale": [],
        "iterate": [],
        "kill": [],
    }
    for _, r in df.sort_values("score", ascending=False).iterrows():
        item = {
            "ad_id": r["name"] or r["ad_name"],
            "name": r["name"] or r["ad_name"],
            "score": float(r["score"]),
            "spend": round(to_float(r["spend"]), 2),
            "roas": round(to_float(r["purchase_roas"]), 2),
            "cpr": round(to_float(r["cpr"]), 2),
            "cvr": round(safe_div(to_float(r["purchases"]), max(to_float(r["clicks"]), 1)), 4),
            "ctr": round(r["ctr"], 4),
            "hook": round(r["hook"], 4),
            "hold": round(r["hold"], 4),
            "tips": r["notes"],
        }
        bucket[r["action"]].append(item)

    summary = {
        "spend": round(spend, 2),
        "purchases": int(round(purchases)),
        "clicks": int(round(to_float(df["clicks"].sum()))),
        "avg_roas": round(avg_roas, 2),
        "settings": s,
    }

    guidance = [
        "Check the biggest leak in the funnel (see diagnosis) before changing budgets.",
        "If ≥72h above target ROAS and (ideally) ≥60% 7-day click, scale +20%. Else wait 24h.",
        "If below breakeven at ≥min test spend, kill or iterate creative and relaunch.",
        "For soft metrics: Hook≥30% and Hold≥10% and CTR≥1.5% are strong early signals.",
    ]

    return {
        "summary": summary,
        "funnel": funnel,
        "watching": winner_names,
        "buckets": bucket,
        "guidance": guidance,
    }

# ----------------- Enrich tips with external data ----------------

async def enrich_with_external(context: Dict[str, Any]) -> List[Dict[str, str]]:
    """Create queries from the biggest leaks / weak metrics and fetch snippets."""
    if not EXTERNAL_DATA or not SERP_API_KEY:
        return []

    q = []
    # Funnel-based queries
    f = context.get("funnel", {}).get("blended", {})
    if f:
        if f.get("ic_to_purchase", 1) < 0.45:
            q.append("how to improve checkout conversion trust signals older shoppers")
            q.append("ecommerce checkout best practices 1 field email")
        if f.get("click_to_atc", 1) < 0.20:
            q.append("product page conversion rate improve above the fold proof price anchoring")
    # Audience/creative soft signals
    # (We keep it generic—your prompts tab handles audience specifics)
    q.append("facebook ads CTR benchmarks ecommerce 2024")
    q.append("best practices video ad hook first 3 seconds DTC")

    return await serp_snippets(list(dict.fromkeys(q)))  # dedupe while preserving order

def external_to_bullets(snips: List[Dict[str, str]]) -> List[str]:
    out = []
    for s in snips[:6]:
        line = f"• {s.get('title','').strip()}: {s.get('snippet','').strip()}"
        out.append(line[:260])
    return out

# --------------------------- Endpoints --------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    # Simple dark UI with tabs (ingest/analyze/coach/prompt/script)
    # Keep structure similar to the version you liked.
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Erto Agent</title>
<style>
  :root { --bg:#0b1217; --card:#121a21; --muted:#94a3b8; --text:#e2e8f0; --accent:#22d3ee; }
  body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui, -apple-system, Segoe UI, Roboto, Inter, sans-serif}
  .wrap{max-width:1100px;margin:24px auto;padding:0 16px}
  .tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
  .tab{padding:10px 14px;border-radius:10px;background:#0f1620;border:1px solid #1e293b;cursor:pointer}
  .tab.active{background:var(--card);border-color:#334155}
  .card{background:var(--card);border:1px solid #1f2a37;border-radius:14px;padding:16px;margin-bottom:16px;box-shadow:0 0 0 1px rgba(255,255,255,0.02) inset}
  .row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
  .row2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .bubble{display:inline-block;background:#0e1720;border:1px solid #223042;padding:6px 10px;border-radius:999px;margin:2px 6px 6px 0}
  .muted{color:var(--muted)}
  textarea,input,button{width:100%;border-radius:10px;border:1px solid #233143;background:#0c131a;color:var(--text);padding:10px}
  button{cursor:pointer;background:#0b3942;border-color:#0e4a57}
  code{background:#0e1720;padding:2px 6px;border-radius:6px;border:1px solid #223042}
  .kpi{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
  .kpi .card{padding:12px}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid #2a3a4b;background:#0b141c}
  .ad{border:1px solid #223042;border-radius:12px;padding:12px;margin-bottom:10px;background:#0e1720}
  .ad h4{margin:2px 0 8px 0}
  .small{font-size:12px}
</style>
</head>
<body>
<div class="wrap">
  <div class="tabs">
    <div class="tab active" data-s="#ingest">Ingest</div>
    <div class="tab" data-s="#analyze">Analyze</div>
    <div class="tab" data-s="#coach">Coach</div>
    <div class="tab" data-s="#prompt">Prompt Lab</div>
    <div class="tab" data-s="#script">Script Doctor</div>
  </div>

  <div id="ingest" class="card">
    <h3>Upload Meta CSV</h3>
    <input id="csvfile" type="file" accept=".csv"/>
    <div style="margin-top:8px"><button onclick="sendCSV()">Ingest</button></div>
    <pre id="ingest_out" class="muted small"></pre>
  </div>

  <div id="analyze" class="card" style="display:none">
    <h3>Account Analyzer</h3>
    <button onclick="runAnalyze()">Run analyze</button>
    <div id="an_out"></div>
  </div>

  <div id="coach" class="card" style="display:none">
    <h3>Coach Q&A</h3>
    <input id="coach_q" placeholder="Ask: 'What’s my ROAS and should I scale?'" />
    <div style="margin-top:8px"><button onclick="askCoach()">Ask</button></div>
    <div id="coach_out"></div>
  </div>

  <div id="prompt" class="card" style="display:none">
    <h3>Prompt Lab</h3>
    <div class="row2">
      <input id="p_goal" placeholder="Goal (e.g., generate 10 UGC hooks for VSL)"/>
      <input id="p_context" placeholder="Context (e.g., 55+ women, feel young & seen)"/>
    </div>
    <div style="margin-top:8px"><button onclick="makePrompt()">Make Prompt</button></div>
    <pre id="p_out" class="small"></pre>
  </div>

  <div id="script" class="card" style="display:none">
    <h3>Script Doctor</h3>
    <textarea id="script_text" rows="10" placeholder="Paste your script..."></textarea>
    <div style="margin-top:8px"><button onclick="doctor()">Analyze Script</button></div>
    <pre id="s_out" class="small"></pre>
  </div>
</div>

<script>
  document.querySelectorAll('.tab').forEach(t=>{
    t.onclick = ()=>{
      document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
      t.classList.add('active');
      const sel = t.getAttribute('data-s');
      document.querySelectorAll('.card').forEach(c=>c.style.display='none');
      document.querySelector(sel).style.display='block';
    };
  });

  async function sendCSV(){
    const f = document.getElementById('csvfile').files[0];
    if(!f){ alert('Choose a CSV first.'); return; }
    const fd = new FormData(); fd.append('file', f);
    const r = await fetch('/ingest_csv', {method:'POST', body:fd});
    document.getElementById('ingest_out').textContent = await r.text();
  }

  function block(title, html){
    return `<div class="card"><h3>${title}</h3>${html}</div>`;
  }
  function pill(k,v){ return `<span class="pill">${k}: <b>${v}</b></span>`; }

  async function runAnalyze(){
    const r = await fetch('/analyze');
    if(!r.ok){ document.getElementById('an_out').innerHTML = '<div class="muted">Upload a CSV first.</div>'; return; }
    const d = await r.json();

    let top = '';
    top += '<div class="kpi">';
    top += `<div class="card"><div class="muted small">Spend</div><div><b>$${d.summary.spend}</b></div></div>`;
    top += `<div class="card"><div class="muted small">Purchases</div><div><b>${d.summary.purchases}</b></div></div>`;
    top += `<div class="card"><div class="muted small">Avg ROAS</div><div><b>${d.summary.avg_roas}</b></div></div>`;
    top += `<div class="card"><div class="muted small">Target ROAS</div><div><b>${d.summary.settings.target_roas}</b></div></div>`;
    top += '</div>';

    const f = d.funnel.blended;
    let fun = '';
    fun += pill('Click→ATC', (f.click_to_atc*100).toFixed(1)+'%')+' ';
    fun += pill('ATC→IC', (f.atc_to_ic*100).toFixed(1)+'%')+' ';
    fun += pill('IC→Purchase', (f.ic_to_purchase*100).toFixed(1)+'%');

    let diag = '';
    d.funnel.diagnosis.forEach(x=>{ diag += `<div class="bubble">${x}</div>`; });

    let watch = '';
    (d.watching||[]).forEach(w=>{ watch += `<div class="bubble">${w}</div>`; });

    function renderAds(arr){
      return arr.map(a=>{
        return `<div class="ad">
          <h4>${a.name || a.ad_id}</h4>
          <div class="muted small">Score ${a.score} • Spend $${a.spend} • ROAS ${a.roas} • CPR $${a.cpr} • CVR ${(a.cvr*100).toFixed(1)}% • CTR ${(a.ctr*100).toFixed(1)}% • Hook ${(a.hook*100).toFixed(1)}% • Hold ${(a.hold*100).toFixed(1)}%</div>
          <div style="margin-top:6px">${(a.tips||[]).map(t=>`<div class="bubble">${t}</div>`).join('')}</div>
        </div>`;
      }).join('');
    }

    let buckets = '';
    buckets += block('Scale', renderAds(d.buckets.scale) || '<div class="muted">—</div>');
    buckets += block('Iterate', renderAds(d.buckets.iterate) || '<div class="muted">—</div>');
    buckets += block('Kill', renderAds(d.buckets.kill) || '<div class="muted">—</div>');

    let guide = '';
    d.guidance.forEach(g=>{ guide += `<div class="bubble">${g}</div>`; });

    let ext = '';
    if(d.external_snippets && d.external_snippets.length){
      ext += '<div class="card"><h3>External nuggets (for tips only)</h3>';
      d.external_snippets.forEach(s=>{
        ext += `<div class="small">• <b>${s.title}</b> — ${s.snippet}</div>`;
      });
      ext += '</div>';
    }

    document.getElementById('an_out').innerHTML =
      block('Summary', top) +
      block('Funnel', fun + '<div style="margin-top:8px">'+diag+'</div>') +
      block('Watching (potential winners)', watch || '<div class="muted">—</div>') +
      buckets +
      block('Playbook', guide) +
      ext;
  }

  async function askCoach(){
    const q = document.getElementById('coach_q').value || 'What should I do next?';
    const r = await fetch('/coach', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question:q})});
    const d = await r.json();
    let html = '';
    html += `<div class="bubble">Answer</div><div class="card"><div>${d.answer}</div></div>`;
    if(d.external_snippets && d.external_snippets.length){
      html += '<div class="card"><div class="muted small">External nuggets used:</div>';
      d.external_snippets.forEach(s=> html += `<div class="small">• <b>${s.title}</b> — ${s.snippet}</div>`);
      html += '</div>';
    }
    document.getElementById('coach_out').innerHTML = html;
  }

  async function makePrompt(){
    const goal = document.getElementById('p_goal').value || '';
    const ctx = document.getElementById('p_context').value || '';
    const r = await fetch('/prompt_lab', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({goal:goal, context:ctx})});
    document.getElementById('p_out').textContent = JSON.stringify(await r.json(), null, 2);
  }

  async function doctor(){
    const txt = document.getElementById('script_text').value || '';
    const r = await fetch('/script_doctor', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({script:txt})});
    document.getElementById('s_out').textContent = JSON.stringify(await r.json(), null, 2);
  }
</script>
</body>
</html>
    """)

# ----------------------- Ingest endpoints -----------------------

@app.post("/ingest_csv")
async def ingest_csv(file: UploadFile = File(...)):
    global LATEST_DF, LAST_ERROR
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
        # Basic cleanups (strip headers)
        df.columns = [str(c).strip() for c in df.columns]
        LATEST_DF = df
        LAST_ERROR = None
        return JSONResponse({"status": "ok", "rows": int(df.shape[0]), "cols": list(df.columns)[:10]})
    except Exception as e:
        LAST_ERROR = f"Ingest error: {e}"
        return JSONResponse({"detail": "Server failed to ingest CSV. Hit /debug/last_error for details."}, status_code=500)

@app.get("/ingest_csv_debug")
async def ingest_csv_debug():
    if LATEST_DF is None:
        return JSONResponse({"detail": "No CSV ingested yet."}, status_code=400)
    df = LATEST_DF
    return JSONResponse({
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "sample": json.loads(df.head(5).fillna("").to_json(orient="records"))
    })

@app.get("/debug/last_error")
async def last_error():
    return {"last_error": LAST_ERROR}

# ----------------------- Analyze endpoint -----------------------

@app.get("/analyze")
async def analyze():
    if LATEST_DF is None:
        return JSONResponse({"detail": "Upload a CSV first."}, status_code=400)

    core = analyze_core(LATEST_DF)

    # External snippets used ONLY for tips/suggestions reasoning
    snips = await enrich_with_external(core) if EXTERNAL_DATA and SERP_API_KEY else []
    core["external_snippets"] = external_to_bullets(snips)

    return JSONResponse(core)

# ------------------------- Coach endpoint -----------------------

@app.post("/coach")
async def coach(payload: Dict[str, Any]):
    question = (payload or {}).get("question", "").strip() or "What should I do next?"

    if LATEST_DF is None:
        return JSONResponse({"answer": "Upload a CSV first so I can read your account."})

    core = analyze_core(LATEST_DF)

    s = core["summary"]["settings"]
    f = core["funnel"]["blended"]

    # A reasoned answer (internal logic only)
    answer_lines = []
    answer_lines.append(
        f"Spend ${core['summary']['spend']:.2f} on {core['summary']['purchases']} purchases → avg ROAS {core['summary']['avg_roas']:.2f} (target {s['target_roas']:.2f})."
    )
    answer_lines.append(
        f"Funnel — Click→ATC {ratio_to_pct(f['click_to_atc'])}%, ATC→IC {ratio_to_pct(f['atc_to_ic'])}%, IC→Purchase {ratio_to_pct(f['ic_to_purchase'])}%."
    )
    if core["funnel"]["diagnosis"]:
        answer_lines.append("Diagnosis: " + "; ".join(core["funnel"]["diagnosis"]))

    # Next actions logic
    if core["summary"]["avg_roas"] >= s["target_roas"]:
        if s.get("assume_click_attrib_ok", False):
            answer_lines.append("You’re above target — scale budgets +20% and monitor daily.")
        else:
            answer_lines.append("If ≥60% of conversions are from 7-day click, scale +20%; otherwise hold and launch 2 creatives.")
    else:
        answer_lines.append("Below target — patch the biggest funnel leak first, then relaunch 2–3 variants with new hooks/angles.")

    # External insights appended to reasoning only
    snips = await enrich_with_external(core) if EXTERNAL_DATA and SERP_API_KEY else []
    return JSONResponse({
        "answer": " ".join(answer_lines),
        "external_snippets": external_to_bullets(snips)
    })

# ----------------------- Prompt Lab endpoint --------------------

@app.post("/prompt_lab")
async def prompt_lab(payload: Dict[str, Any]):
    goal = (payload or {}).get("goal", "").strip()
    context = (payload or {}).get("context", "").strip()

    # A stronger prompt set
    prompt = f"""You are a senior DTC creative strategist.
Goal: {goal or "Generate high-converting creative instructions."}
Context: {context or "Prospecting, broad; optimize for 55+ women if relevant."}

Deliver:
1) 10 hooks (≤8 words) that stop scroll.
2) 3 big ideas (angle + proof).
3) 1 full 45–60s VSL outline (Hook → Problem → Promise → Proof → Mechanism → Offer → CTA).
4) 5 on-screen text lines (≤6 words), timed at 0s/2s/4s/6s/8s.
Style: concise bullets, no fluff. Keep language natural and empathetic for older audiences."""
    return JSONResponse({
        "goal": goal,
        "context": context,
        "prompt": prompt
    })

# ---------------------- Script Doctor endpoint ------------------

SEGMENTS = ["Hook", "Problem", "Agitate", "Mechanism", "Proof", "Offer", "CTA"]

@app.post("/script_doctor")
async def script_doctor(payload: Dict[str, Any]):
    txt = (payload or {}).get("script", "").strip()
    if not txt:
        return JSONResponse({"detail": "Paste a script."}, status_code=400)

    # Super-light heuristic segmentation
    segs = []
    lower = txt.lower()
    if any(w in lower for w in ["how", "what", "why", "?"]): segs.append("Hook")
    if any(w in lower for w in ["pain", "struggle", "tired", "problem"]): segs.append("Problem")
    if any(w in lower for w in ["worse", "frustrat", "decline"]): segs.append("Agitate")
    if any(w in lower for w in ["because", "mechanism", "works", "science", "infrared", "atp"]): segs.append("Mechanism")
    if any(w in lower for w in ["proof", "study", "doctor", "testimonial", "before", "after", "over"]): segs.append("Proof")
    if any(w in lower for w in ["today", "now", "guarantee", "save", "bundle"]): segs.append("Offer")
    if any(w in lower for w in ["shop", "buy", "cta", "tap", "click", "learn more"]): segs.append("CTA")
    if not segs: segs = ["Hook"]

    notes = [
        "Add a crisp 1-line hook with a hard pattern break in the first 2s.",
        "Insert quick proof (testimonial, number, or before/after) by 8–12s.",
        "End with a direct CTA and a reason to act now (micro-urgency).",
        "For 55+ audiences, use larger on-screen text (≥20pt), slower cut pace, and clear voiceover enunciation."
    ]

    return JSONResponse({
        "segments_detected": list(dict.fromkeys(segs)),
        "notes": notes
    })
