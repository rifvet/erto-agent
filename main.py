# main.py
# Media Buying Agent — simple, single-file FastAPI app
# - CSV ingest + debug
# - Analyze: scale/kill/iterate + potential_winners + creative_gaps + funnel doctor
# - Coach: Q&A with optional external data (SERPAPI)
# - Prompt lab: high-value prompt templates
# - Script doctor: segment + feedback
# - Settings + Debug
#
# Requirements (example):
# fastapi==0.111.0
# uvicorn[standard]==0.30.1
# python-dotenv==1.0.1
# pandas==2.2.2
# pydantic==2.8.2
# python-multipart==0.0.9
# httpx==0.27.0
# requests==2.32.3 (bundled via std images, but fine to add)

import io
import os
import json
import math
from typing import Optional, List, Dict, Any

import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

APP_NAME = "Erto Media Agent"

app = FastAPI(title=APP_NAME)

# CORS for quick testing / Swagger
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# In-memory state
# ----------------------------
_last_df: Optional[pd.DataFrame] = None
_last_error: Optional[Dict[str, Any]] = None

class SettingsModel(BaseModel):
    breakeven_roas: float = 1.0            # below this = losing
    target_roas: float = 2.0               # scale when >= this (with other checks)
    min_test_spend: float = 20.0           # spend before we judge harshly
    min_scale_spend: float = 50.0          # spend before we consider true scale
    max_cpr: Optional[float] = None        # optional guardrail (aka CPA on Meta)
    hook_good: float = 0.30                # 30%+
    hold_good: float = 0.10                # 10%+
    ctr_good: float = 0.015                # 1.5%+
    external_data: bool = False            # allow SERP lookups in Coach
    assume_click_attrib_ok: bool = False   # if True, skip the 7d-click ≥60% check note

_settings = SettingsModel()

# ----------------------------
# Utils
# ----------------------------
def _set_last_error(exc: Exception, context: str, payload: Optional[dict] = None):
    global _last_error
    _last_error = {
        "context": context,
        "error": str(exc),
        "payload": payload or {},
        "type": exc.__class__.__name__,
    }

def _to_num(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if pd.isna(x):
            return None
        return float(x)
    try:
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None

def _ratio(n: Optional[float], d: Optional[float]) -> Optional[float]:
    n = _to_num(n)
    d = _to_num(d)
    if n is None or d is None or d == 0:
        return None
    return n / d

def _pct_to_rate(x: Optional[float]) -> Optional[float]:
    """Facebook exports CTR etc often as a percent (e.g., 3.5). Convert to 0.035 if >1."""
    v = _to_num(x)
    if v is None:
        return None
    return v / 100.0 if v > 1 else v

def _read_csv_bytes(file: UploadFile) -> pd.DataFrame:
    # robust to utf-8-sig and commas
    raw = file.file.read()
    try:
        txt = raw.decode("utf-8-sig", errors="ignore")
    except Exception:
        txt = raw.decode("latin-1", errors="ignore")
    buf = io.StringIO(txt)
    df = pd.read_csv(buf)
    return df

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # lower/strip names
    rename = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=rename)

    # map common names to internal schema
    mapping = {
        "ad name": "ad_name",
        "ad set name": "adset_name",
        "campaign name": "campaign_name",
        "day": "day",
        "amount spent (usd)": "spend",
        "purchases": "purchases",
        "website purchases": "purchases_website",
        "purchase roas (return on ad spend)": "roas",
        "website purchase roas (return on advertising spend)": "roas",
        "cost per purchase": "cpr",
        "cost per result": "cpr",  # treat CPR as CPA
        "ctr (link click-through rate)": "ctr",
        "link clicks": "link_clicks",
        "cpc (cost per link click)": "cpc",
        "adds to cart": "atc",
        "website adds to cart": "atc",
        "checkouts initiated": "ic",
        "website checkouts initiated": "ic",
        "cost per add to cart": "cpatc",
        "cost per checkout initiated": "cpic",
        "reach": "reach",
        "frequency": "frequency",
        "cpm (cost per 1,000 impressions)": "cpm",
        "hook rate": "hook_rate",
        "video hook": "hook_rate",
        "hold rate": "hold_rate",
        "video hold": "hold_rate",
        "video average play time": "avg_play_time",
        "reporting starts": "reporting_starts",
        "reporting ends": "reporting_ends",
    }

    for k, v in list(mapping.items()):
        if k in df.columns:
            df[v] = df[k]

    # ensure required id/name
    if "ad_name" not in df.columns:
        # fallbacks
        for cand in ["ad", "ad id", "ad_id", "adname"]:
            if cand in df.columns:
                df["ad_name"] = df[cand]
                break
    if "ad_name" not in df.columns:
        df["ad_name"] = df.get("ad set name", df.get("campaign name", "NA"))

    # numeric coercions
    num_cols = [
        "spend","purchases","purchases_website","cpr","roas","ctr","link_clicks",
        "cpc","atc","ic","cpatc","cpic","reach","frequency","cpm",
        "hook_rate","hold_rate","avg_play_time"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # CTR and Hook/Hold are often percents in exports — normalize to rates (0–1)
    for c in ["ctr", "hook_rate", "hold_rate"]:
        if c in df.columns:
            df[c] = df[c].apply(_pct_to_rate)

    # unify purchases
    if "purchases" not in df.columns and "purchases_website" in df.columns:
        df["purchases"] = df["purchases_website"]
    elif "purchases" in df.columns and "purchases_website" in df.columns:
        # prefer website purchases when both are present and website is not NaN
        df["purchases"] = df["purchases_website"].fillna(df["purchases"])

    # make sure spend exists
    if "spend" not in df.columns:
        # try "amount spent" without (USD)
        for cand in ["amount spent", "spend"]:
            if cand in df.columns:
                df["spend"] = pd.to_numeric(df[cand], errors="coerce")
                break

    # derive cvr (purchases / clicks) if possible
    if "link_clicks" in df.columns and "purchases" in df.columns:
        with pd.option_context("mode.use_inf_as_na", True):
            df["cvr"] = pd.to_numeric(df["purchases"], errors="coerce") / pd.to_numeric(df["link_clicks"], errors="coerce")
    else:
        df["cvr"] = pd.NA

    # derive effective cpr (aka CPA) if not provided
    if "cpr" not in df.columns:
        if "spend" in df.columns and "purchases" in df.columns:
            df["cpr"] = pd.to_numeric(df["spend"], errors="coerce") / pd.to_numeric(df["purchases"], errors="coerce")
        else:
            df["cpr"] = pd.NA

    return df

def _aggregate_by_ad(df: pd.DataFrame) -> pd.DataFrame:
    # Sum spend/clicks/purchases; average soft metrics by spend-weight (simple: mean)
    agg = {
        "spend": "sum",
        "purchases": "sum",
        "link_clicks": "sum",
        "atc": "sum",
        "ic": "sum",
        "cpr": "mean",
        "roas": "mean",
        "ctr": "mean",
        "hook_rate": "mean",
        "hold_rate": "mean",
        "cpm": "mean",
        "reach": "sum",
        "frequency": "mean",
        "cvr": "mean",
    }
    cols = [c for c in agg if c in df.columns]
    g = df.groupby("ad_name", dropna=False)[cols].agg(agg).reset_index()

    # recompute stronger metrics post-agg
    if "spend" in g.columns and "purchases" in g.columns:
        g["cpr_eff"] = g.apply(lambda r: _ratio(r["spend"], r["purchases"]), axis=1)
        g["cpr"] = g["cpr"].fillna(g["cpr_eff"])
    if "spend" in g.columns and "roas" in g.columns:
        # keep as mean ROAS across days; OK for directional
        pass
    if "link_clicks" in g.columns and "purchases" in g.columns:
        g["cvr"] = g.apply(lambda r: _ratio(r["purchases"], r["link_clicks"]), axis=1)
    return g

def _creative_gaps(df_ad: pd.DataFrame) -> Dict[str, Any]:
    # Simple suggestion count: if #ads with spend>min_test < 6, say we need (6 - winners)
    active = df_ad[df_ad["spend"] >= _settings.min_test_spend]
    needed = max(0, 6 - len(active))
    # Angle mix hint: if hooks good but CVR low -> 'proof'/'social'; if CTR low -> 'curiosity/pain'
    angle_mix = {}
    ctr_low = (df_ad["ctr"].fillna(0) < _settings.ctr_good).mean() if "ctr" in df_ad else 0.0
    hold_low = (df_ad["hold_rate"].fillna(0) < _settings.hold_good).mean() if "hold_rate" in df_ad else 0.0
    cvr_low = (df_ad["cvr"].fillna(0) < 0.02).mean() if "cvr" in df_ad else 0.0  # 2% rough
    # Distribute weights
    angle_mix["pain"] = round(min(50, 20 + 40*ctr_low))
    angle_mix["curiosity"] = round(min(50, 15 + 35*ctr_low))
    angle_mix["proof"] = round(min(50, 15 + 35*cvr_low))
    angle_mix["social"] = round(min(30, 10 + 20*(hold_low>0.5)))
    # normalize to 100
    s = sum(angle_mix.values()) or 1
    for k in angle_mix:
        angle_mix[k] = int(round(100*angle_mix[k]/s))
    return {"needed_new_creatives": needed, "angle_mix": angle_mix, "bans": []}

def _funnel_doctor(df_ad: pd.DataFrame) -> Dict[str, Any]:
    # Compute blended funnel and diagnose biggest leak
    clicks = df_ad["link_clicks"].sum() if "link_clicks" in df_ad else 0
    atc = df_ad["atc"].sum() if "atc" in df_ad else 0
    ic = df_ad["ic"].sum() if "ic" in df_ad else 0
    purchases = df_ad["purchases"].sum() if "purchases" in df_ad else 0

    def safe_rate(n, d):
        return float(n)/float(d) if d and n is not None else 0.0

    r_click_to_atc = safe_rate(atc, clicks)
    r_atc_to_ic = safe_rate(ic, atc)
    r_ic_to_purchase = safe_rate(purchases, ic)

    diagnosis = []
    if r_click_to_atc < 0.05:
        diagnosis.append("Low Click→ATC (possible LP trust/clarity issues, offer mismatch, slow load).")
    if r_atc_to_ic < 0.40 and atc >= 10:
        diagnosis.append("Low ATC→Checkout (shipping sticker shock, weak urgency, cart friction).")
    if r_ic_to_purchase < 0.50 and ic >= 10:
        diagnosis.append("Low Checkout→Purchase (checkout UX friction, payment issues, reassurance missing).")
    if not diagnosis:
        diagnosis.append("Funnel is relatively healthy; improvements likely from creative/targeting or AOV uplift.")

    return {
        "blended": {
            "clicks": int(clicks),
            "add_to_cart": int(atc),
            "initiate_checkout": int(ic),
            "purchases": int(purchases),
            "click_to_atc": round(r_click_to_atc, 4),
            "atc_to_ic": round(r_atc_to_ic, 4),
            "ic_to_purchase": round(r_ic_to_purchase, 4),
        },
        "diagnosis": diagnosis,
    }

def _serp_search(q: str, num: int = 5) -> List[Dict[str, str]]:
    if not _settings.external_data:
        return []
    api_key = os.getenv("SERPAPI_KEY", "").strip()
    if not api_key:
        return [{"source": "SERPAPI", "title": "External data is OFF or missing SERPAPI_KEY", "link": "", "snippet": ""}]
    try:
        params = {
            "engine": "google",
            "q": q,
            "num": num,
            "api_key": api_key,
        }
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        out = []
        for item in data.get("organic_results", [])[:num]:
            out.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "SERPAPI",
            })
        return out
    except Exception as e:
        _set_last_error(e, "serp_search", {"q": q})
        return [{"source": "SERPAPI", "title": "Lookup failed", "link": "", "snippet": str(e)}]

# ----------------------------
# Schemas
# ----------------------------
class AnalyzeResponse(BaseModel):
    scale: List[Dict[str, Any]]
    kill: List[Dict[str, Any]]
    iterate: List[Dict[str, Any]]
    potential_winners: List[Dict[str, Any]]
    creative_gaps: Dict[str, Any]
    funnel: Dict[str, Any]

class CoachRequest(BaseModel):
    question: str

class PromptRequest(BaseModel):
    goal: str
    context: Optional[str] = None

class ScriptRequest(BaseModel):
    script: str

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    # Minimal dark UI with tabs; uses fetch to hit endpoints.
    html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Erto Media Agent</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
:root{color-scheme:dark;}
body{margin:0;background:#0b0f14;color:#e6edf3;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Helvetica Neue","Noto Sans",Arial,"Apple Color Emoji","Segoe UI Emoji";}
header{padding:16px 20px;border-bottom:1px solid #1f2a34;background:#0f141a;display:flex;gap:12px;align-items:center}
h1{font-size:16px;margin:0;color:#8ab4f8;}
small{opacity:.7}
.container{padding:20px;max-width:1100px;margin:0 auto;}
.tabs{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.tab{padding:8px 12px;border:1px solid #1f2a34;border-radius:10px;background:#111821;cursor:pointer}
.tab.active{background:#12263a;border-color:#204363}
.card{background:#0f141a;border:1px solid #1f2a34;border-radius:14px;padding:16px;margin-bottom:16px}
input,textarea,button,select{background:#0b1117;border:1px solid #1f2a34;color:#e6edf3;border-radius:10px;padding:8px}
button{cursor:pointer}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
pre{white-space:pre-wrap;word-break:break-word}
label{font-size:12px;opacity:.8}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid #204363;background:#0e1a26;color:#8ab4f8;font-size:12px}
</style>
</head>
<body>
<header>
  <h1>Erto Media Agent</h1>
  <small>Ingest · Analyze · Coach · Prompt Lab · Script Doctor</small>
  <span id="ext" class="badge" title="External data via SERP">External: OFF</span>
</header>
<div class="container">
  <div class="tabs">
    <div class="tab active" data-t="ingest">Ingest</div>
    <div class="tab" data-t="analyze">Analyze</div>
    <div class="tab" data-t="coach">Coach</div>
    <div class="tab" data-t="prompt">Prompt Lab</div>
    <div class="tab" data-t="script">Script Doctor</div>
    <div class="tab" data-t="settings">Settings</div>
    <div class="tab" data-t="debug">Debug</div>
  </div>

  <div id="ingest" class="card">
    <h3>Upload CSV</h3>
    <div class="row">
      <input type="file" id="csv" />
      <button onclick="ingest()">Ingest</button>
      <button onclick="ingestDebug()">Parse-only</button>
    </div>
    <pre id="ingest_out"></pre>
  </div>

  <div id="analyze" class="card" style="display:none">
    <h3>Analyze</h3>
    <div class="row">
      <button onclick="runAnalyze()">Run analysis</button>
    </div>
    <pre id="analyze_out"></pre>
  </div>

  <div id="coach" class="card" style="display:none">
    <h3>Coach — Ask me anything</h3>
    <div class="row"><input id="q" placeholder="e.g., CTR good but ATC awful — what should I fix?" style="flex:1" />
    <button onclick="ask()">Ask</button></div>
    <pre id="coach_out"></pre>
  </div>

  <div id="prompt" class="card" style="display:none">
    <h3>Prompt Lab</h3>
    <div class="grid">
      <div><label>Goal</label><input id="goal" placeholder="e.g., Hook ideas for UGC video"/></div>
      <div><label>Context (optional)</label><input id="ctx" placeholder="product, audience, objections, etc"/></div>
    </div>
    <div class="row" style="margin-top:8px"><button onclick="makePrompt()">Generate</button></div>
    <pre id="prompt_out"></pre>
  </div>

  <div id="script" class="card" style="display:none">
    <h3>Script Doctor</h3>
    <textarea id="script_txt" rows="8" placeholder="Paste your script here..."></textarea>
    <div class="row" style="margin-top:8px">
      <button onclick="scriptAnalyze()">Analyze</button>
    </div>
    <pre id="script_out"></pre>
  </div>

  <div id="settings" class="card" style="display:none">
    <h3>Settings</h3>
    <div class="grid">
      <div><label>Breakeven ROAS</label><input id="s_breakeven" value="1.0"/></div>
      <div><label>Target ROAS</label><input id="s_target" value="2.0"/></div>
      <div><label>Min Test Spend ($)</label><input id="s_min_test" value="20"/></div>
      <div><label>Min Scale Spend ($)</label><input id="s_min_scale" value="50"/></div>
      <div><label>Max CPR (optional)</label><input id="s_max_cpr" value=""/></div>
      <div><label>Hook good ≥</label><input id="s_hook" value="0.3"/></div>
      <div><label>Hold good ≥</label><input id="s_hold" value="0.1"/></div>
      <div><label>CTR good ≥</label><input id="s_ctr" value="0.015"/></div>
    </div>
    <div class="row" style="margin-top:10px">
      <button onclick="saveSettings(false)">Save</button>
      <button onclick="toggleExternal()">Toggle External Data</button>
    </div>
    <pre id="settings_out"></pre>
  </div>

  <div id="debug" class="card" style="display:none">
    <h3>Debug</h3>
    <div class="row"><button onclick="lastError()">Show last error</button></div>
    <pre id="debug_out"></pre>
  </div>
</div>

<script>
const tabs = document.querySelectorAll('.tab');
tabs.forEach(t => t.addEventListener('click', () => {
  tabs.forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  const id = t.getAttribute('data-t');
  document.querySelectorAll('.card').forEach(c => c.style.display = 'none');
  document.getElementById(id).style.display = 'block';
}));

async function ingest(){
  const f = document.getElementById('csv').files[0];
  if(!f){ alert('Pick a CSV first'); return;}
  const fd = new FormData(); fd.append('file', f);
  const r = await fetch('/ingest_csv', {method:'POST', body:fd});
  document.getElementById('ingest_out').textContent = await r.text();
}
async function ingestDebug(){
  const f = document.getElementById('csv').files[0];
  if(!f){ alert('Pick a CSV first'); return;}
  const fd = new FormData(); fd.append('file', f);
  const r = await fetch('/ingest_csv_debug', {method:'POST', body:fd});
  document.getElementById('ingest_out').textContent = await r.text();
}
async function runAnalyze(){
  const r = await fetch('/analyze'); 
  document.getElementById('analyze_out').textContent = await r.text();
}
async function ask(){
  const q = document.getElementById('q').value || '';
  const r = await fetch('/coach', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:q})});
  document.getElementById('coach_out').textContent = await r.text();
}
async function makePrompt(){
  const goal = document.getElementById('goal').value || '';
  const ctx = document.getElementById('ctx').value || '';
  const r = await fetch('/prompt', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({goal, context:ctx})});
  document.getElementById('prompt_out').textContent = await r.text();
}
async function scriptAnalyze(){
  const script = document.getElementById('script_txt').value || '';
  const r = await fetch('/script/analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({script})});
  document.getElementById('script_out').textContent = await r.text();
}
async function saveSettings(toggle){
  const body = {
    breakeven_roas: parseFloat(document.getElementById('s_breakeven').value),
    target_roas: parseFloat(document.getElementById('s_target').value),
    min_test_spend: parseFloat(document.getElementById('s_min_test').value),
    min_scale_spend: parseFloat(document.getElementById('s_min_scale').value),
    max_cpr: document.getElementById('s_max_cpr').value ? parseFloat(document.getElementById('s_max_cpr').value) : null,
    hook_good: parseFloat(document.getElementById('s_hook').value),
    hold_good: parseFloat(document.getElementById('s_hold').value),
    ctr_good: parseFloat(document.getElementById('s_ctr').value),
  };
  const r = await fetch('/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  document.getElementById('settings_out').textContent = await r.text();
  await getSettings();
}
async function getSettings(){
  const r = await fetch('/settings');
  const s = await r.json();
  document.getElementById('s_breakeven').value = s.breakeven_roas;
  document.getElementById('s_target').value = s.target_roas;
  document.getElementById('s_min_test').value = s.min_test_spend;
  document.getElementById('s_min_scale').value = s.min_scale_spend;
  document.getElementById('s_max_cpr').value = s.max_cpr ?? '';
  document.getElementById('s_hook').value = s.hook_good;
  document.getElementById('s_hold').value = s.hold_good;
  document.getElementById('s_ctr').value = s.ctr_good;
  document.getElementById('ext').textContent = 'External: ' + (s.external_data ? 'ON' : 'OFF');
}
async function toggleExternal(){
  const r = await fetch('/settings/toggle_external', {method:'POST'});
  document.getElementById('settings_out').textContent = await r.text();
  await getSettings();
}
getSettings();
</script>
</body>
</html>
"""
    return HTMLResponse(html)

@app.post("/ingest_csv_debug")
def ingest_csv_debug(file: UploadFile = File(...)):
    try:
        df = _read_csv_bytes(file)
        info = {
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "sample": df.fillna("").head(5).to_dict(orient="records"),
        }
        return JSONResponse(info)
    except Exception as e:
        _set_last_error(e, "ingest_csv_debug")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest_csv")
def ingest_csv(file: UploadFile = File(...)):
    global _last_df
    try:
        df = _read_csv_bytes(file)
        df = _normalize_columns(df)
        _last_df = df
        return PlainTextResponse("OK — CSV ingested. Rows: {}".format(len(df)))
    except Exception as e:
        _set_last_error(e, "ingest_csv")
        raise HTTPException(status_code=500, detail="Server failed to ingest CSV. Hit /debug/last_error for details.")

@app.get("/analyze", response_model=AnalyzeResponse)
def analyze():
    if _last_df is None or _last_df.empty:
        raise HTTPException(status_code=400, detail="No data ingested yet. Upload a CSV first.")

    try:
        df = _last_df.copy()
        df_ad = _aggregate_by_ad(df)

        results = {"scale": [], "kill": [], "iterate": [], "potential_winners": []}

        for _, r in df_ad.iterrows():
            ad = str(r["ad_name"])
            spend = _to_num(r.get("spend"))
            roas = _to_num(r.get("roas"))
            cpr = _to_num(r.get("cpr"))
            cvr = _to_num(r.get("cvr"))
            ctr = _to_num(r.get("ctr"))
            hook = _to_num(r.get("hook_rate"))
            hold = _to_num(r.get("hold_rate"))
            purchases = _to_num(r.get("purchases"))

            # default reasons
            reason = ""
            bucket = "iterate"

            # Kill: enough spend but below breakeven or CPR too high / zero results
            if spend and spend >= _settings.min_test_spend:
                if (roas is not None and roas < _settings.breakeven_roas) or (purchases is None or purchases == 0):
                    bucket = "kill"
                    reason = "Below breakeven or no purchases at test spend"
                if _settings.max_cpr is not None and (cpr and cpr > _settings.max_cpr):
                    bucket = "kill"
                    reason = "CPA (CPR) above limit"

            # Scale: enough spend and ≥ target roas
            if spend and spend >= _settings.min_scale_spend and roas is not None and roas >= _settings.target_roas:
                bucket = "scale"
                reason = "≥ target ROAS and enough spend"
                if not _settings.assume_click_attrib_ok:
                    reason += " (Verify ≥60% from 7d-click before scaling again)"

            # Iterate: not enough spend
            if spend is None or spend < _settings.min_test_spend:
                bucket = "iterate"
                reason = f"Not enough spend (<{_settings.min_test_spend})"

            item = {
                "ad_id": ad,
                "name": ad,
                "spend": round(spend or 0, 2),
                "roas": round(roas or 0, 2) if roas is not None else 0,
                "cpr": round(cpr or 0, 2) if cpr is not None else 0,
                "cvr": round(cvr or 0, 4) if cvr is not None else 0,
                "ctr": round(ctr or 0, 4) if ctr is not None else 0,
                "hook": round(hook or 0, 4) if hook is not None else 0,
                "hold": round(hold or 0, 4) if hold is not None else 0,
                "reason": reason,
            }
            results[bucket].append(item)

            # Potential winners: in test band with strong soft signals
            test_upper = _settings.min_test_spend
            test_lower = max(5.0, 0.25 * _settings.min_test_spend)
            soft_good = ((hook or 0) >= _settings.hook_good) or ((hold or 0) >= _settings.hold_good) or ((ctr or 0) >= _settings.ctr_good)
            if spend and test_lower <= spend < test_upper and soft_good and (purchases or 0) == 0:
                results["potential_winners"].append({
                    "ad_id": ad,
                    "name": ad,
                    "spend": round(spend, 2),
                    "soft_signals": {
                        "hook_ok": bool((hook or 0) >= _settings.hook_good),
                        "hold_ok": bool((hold or 0) >= _settings.hold_good),
                        "ctr_ok": bool((ctr or 0) >= _settings.ctr_good),
                    },
                    "suggestion": "Let it cook or duplicate into fresh DCT. Consider iterating first 3s and CTA.",
                })

        creative_gaps = _creative_gaps(df_ad)
        funnel = _funnel_doctor(df_ad)

        return AnalyzeResponse(
            scale=results["scale"],
            kill=results["kill"],
            iterate=results["iterate"],
            potential_winners=results["potential_winners"],
            creative_gaps=creative_gaps,
            funnel=funnel,
        )
    except Exception as e:
        _set_last_error(e, "analyze")
        raise HTTPException(status_code=500, detail="Analysis failed. Check /debug/last_error")

@app.post("/coach")
def coach(req: CoachRequest):
    if _last_df is None or _last_df.empty:
        raise HTTPException(status_code=400, detail="No data ingested yet.")
    try:
        df_ad = _aggregate_by_ad(_last_df)
        # quick context summary
        spend = float(df_ad["spend"].sum()) if "spend" in df_ad else 0.0
        purchases = float(df_ad["purchases"].sum()) if "purchases" in df_ad else 0.0
        clicks = float(df_ad["link_clicks"].sum()) if "link_clicks" in df_ad else 0.0
        roas_mean = float(df_ad["roas"].mean()) if "roas" in df_ad else 0.0

        funnel = _funnel_doctor(df_ad)
        ext = []
        # Basic “outside” pulls keyed to the question
        if _settings.external_data:
            q = req.question.lower()
            terms = []
            if "ctr" in q or "click" in q:
                terms.append("good facebook ads CTR benchmark ecommerce 2025")
            if "atc" in q or "add to cart" in q:
                terms.append("facebook ads add to cart rate ecommerce benchmarks 2025")
            if "checkout" in q or "purchase" in q or "roas" in q:
                terms.append("ecommerce checkout conversion rate benchmarks 2025")
            if not terms:
                terms.append("meta ads performance benchmarks ecommerce 2025")
            for t in terms:
                ext += _serp_search(t, num=5)

        answer = {
            "summary": {
                "spend": round(spend,2),
                "purchases": int(purchases),
                "clicks": int(clicks),
                "avg_roas": round(roas_mean,2),
                "settings": _settings.model_dump(),
            },
            "funnel": funnel,
            "guidance": [
                "Check the biggest leak in the funnel (see diagnosis) before changing budgets.",
                "If ≥72h above target ROAS and (ideally) ≥60% 7d-click, scale +20%. Else wait 24h.",
                "If below breakeven at ≥min test spend, kill or iterate creative and relaunch.",
                "For soft metrics: Hook≥30% and Hold≥10% and CTR≥1.5% are strong early signals."
            ],
            "external_snippets": ext[:8] if ext else [],
            "answer": f"My take: {req.question}",
        }
        return JSONResponse(answer)
    except Exception as e:
        _set_last_error(e, "coach", {"q": req.question})
        raise HTTPException(status_code=500, detail="Coach failed. See /debug/last_error")

@app.post("/prompt")
def prompt(req: PromptRequest):
    # high-value prompt starter packs
    goal = (req.goal or "").strip().lower()
    ctx = (req.context or "").strip()

    templates = {
        "ugc hooks": "You are a DTC creative strategist. Generate 10 UGC hooks (≤8 words) that stop scroll for {context}. Use pain, curiosity, and proof angles. Output as a simple list.",
        "angles": "You are a performance marketer. Propose 8 creative angles for {context}: 3 pain, 3 curiosity, 2 proof. For each: hook line + CTA suggestion.",
        "lp audit": "Act as a CRO consultant. Audit the landing page for {context}. Give checks for trust, clarity, load speed, social proof, and checkout friction. End with a prioritized action list.",
        "ad brief": "Write a concise ad brief for {context}: goal, audience, big idea, promise, key proof, 3 hooks, 1 CTA, do/don’t list.",
        "headline variants": "Generate 20 headline variations for {context} mixing benefit-led, specificity, and curiosity patterns.",
    }
    key = "ugc hooks"
    if "angle" in goal: key = "angles"
    elif "audit" in goal or "lp" in goal: key = "lp audit"
    elif "brief" in goal: key = "ad brief"
    elif "headline" in goal: key = "headline variants"

    prompt_text = templates[key].format(context=ctx or "the product and audience")
    return JSONResponse({"goal": goal, "template_used": key, "prompt": prompt_text})

@app.post("/script/analyze")
def script_analyze(req: ScriptRequest):
    text = (req.script or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Provide a script.")

    # very lightweight segmentation by cues/keywords
    lower = text.lower()
    segs = []
    def add(name, present):
        if present:
            segs.append(name)

    add("Hook (first 3–5s)", any(k in lower for k in ["hook", "wait", "stop scrolling", "what if", "did you know", "psst"]))
    add("Problem", any(k in lower for k in ["problem", "struggle", "tired of", "frustrated", "sick of"]))
    add("Agitate", any(k in lower for k in ["worse", "costly", "harder", "pain", "risk"]))
    add("Solution", any(k in lower for k in ["we built", "introducing", "here's how", "solution"]))
    add("Proof", any(k in lower for k in ["results", "proof", "study", "before", "after", "testimonial", "guarantee"]))
    add("Offer", any(k in lower for k in ["today only", "% off", "free", "bundle", "bonus"]))
    add("CTA", any(k in lower for k in ["buy now", "shop now", "get yours", "learn more", "tap"]))

    tips = []
    if "hook" not in [s.lower() for s in segs]:
        tips.append("Add a crisp 1-line hook with a hard pattern break in the first 2s.")
    if "Proof" not in segs:
        tips.append("Insert quick proof (testimonial, number, before/after) within 8–12s.")
    if "CTA" not in segs:
        tips.append("End with a direct CTA and a reason to act now (micro-urgency).")
    if len(text.split()) > 180:
        tips.append("Tighten pacing; aim for ~120–150 words for a 45–60s cut.")

    return JSONResponse({
        "segments_detected": segs,
        "notes": tips or ["Solid structure — consider testing a stronger first 3 seconds."],
    })

# ----------------------------
# Settings & Debug
# ----------------------------
@app.get("/settings")
def get_settings():
    return JSONResponse(_settings.model_dump())

@app.post("/settings")
def set_settings(s: SettingsModel):
    global _settings
    _settings = s
    return JSONResponse({"ok": True, "settings": _settings.model_dump()})

@app.post("/settings/toggle_external")
def toggle_external():
    _settings.external_data = not _settings.external_data
    return JSONResponse({"external_data": _settings.external_data})

@app.get("/debug/last_error")
def last_error():
    if not _last_error:
        return PlainTextResponse("No errors recorded.")
    return JSONResponse(_last_error)

# ----------------------------
# Healthcheck (Render)
# ----------------------------
@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")
