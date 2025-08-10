import io
import os
import json
import traceback
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# ---------------------------
# App & settings
# ---------------------------
APP_TITLE = "Erto Agent"
APP_VERSION = "0.6.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple SQLite (file on disk). If you have a DATABASE_URL, the app will use it.
DB_URL = os.environ.get("DATABASE_URL", "sqlite:///metrics.db")
engine = create_engine(DB_URL, future=True)

LAST_ERROR: str = ""

# ---------------------------
# Helpers
# ---------------------------

def set_last_error(e: Exception):
    global LAST_ERROR
    LAST_ERROR = "".join(traceback.format_exception(type(e), e, e.__traceback__))

def read_uploaded_csv(file: UploadFile) -> pd.DataFrame:
    # Read the bytes and let pandas sniff; Meta exports often contain BOM and mixed types
    raw_bytes = file.file.read()
    if not raw_bytes:
        raise ValueError("Empty file upload.")

    bio = io.BytesIO(raw_bytes)
    try:
        df = pd.read_csv(
            bio,
            dtype=str,            # read everything as text first (avoids dtype conflicts)
            keep_default_na=False # keep empty strings rather than NaN initially
        )
    except Exception:
        # Some Meta exports are semicolon-delimited; try again
        bio.seek(0)
        df = pd.read_csv(bio, dtype=str, keep_default_na=False, sep=";")

    # Trim header whitespace
    df.columns = [c.strip() for c in df.columns]
    return df


def coerce_numeric(s: pd.Series) -> pd.Series:
    # Remove commas, percent signs, currency, and cast
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("$", "", regex=False)
         .str.replace("%", "", regex=False)
         .str.strip(),
        errors="coerce"
    )


def pct_to_ratio(series: pd.Series) -> pd.Series:
    """Meta sometimes reports CTR etc. as percentages (e.g., 3.2) not 0.032.
       Heuristic: values > 1.5 are likely percentages. Convert to 0-1 ratios.
    """
    x = coerce_numeric(series)
    return pd.Series(
        [v/100.0 if pd.notna(v) and v > 1.5 else v for v in x],
        index=series.index
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map many possible Meta column names into a canonical schema we use downstream."""
    colmap = {
        # idents
        "ad name": "ad_name",
        "ad set name": "adset_name",
        "adset name": "adset_name",
        "campaign name": "campaign_name",

        # dates
        "day": "dte",
        "reporting starts": "reporting_starts",
        "reporting ends": "reporting_ends",

        # delivery
        "delivery status": "delivery_status",
        "delivery level": "delivery_level",

        # spend metrics
        "amount spent (usd)": "spend",
        "purchase roas (return on ad spend)": "roas_purchase",
        "website purchase roas (return on advertising spend)": "roas_purchase_web",
        "cost per purchase": "cpr_purchase",
        "cost per result": "cpr",  # Meta CPR

        # engagement funnel
        "ctr (link click-through rate)": "ctr",
        "link clicks": "clicks",
        "adds to cart": "atc",
        "website adds to cart": "atc_web",
        "checkouts initiated": "checkout",
        "website checkouts initiated": "checkout_web",
        "purchases": "purchases",
        "website purchases": "purchases_web",

        # cost/impressions
        "cpc (cost per link click)": "cpc",
        "cpm (cost per 1,000 impressions)": "cpm",
        "reach": "reach",
        "frequency": "frequency",

        # creative quality
        "hook rate": "hook_rate",
        "hold rate": "hold_rate",
        "video average play time": "avg_play_time",
    }

    lower = {c.lower(): c for c in df.columns}
    canonical: Dict[str, str] = {}
    for k, v in colmap.items():
        if k in lower:
            canonical[lower[k]] = v

    df2 = df.rename(columns=canonical).copy()

    # Make sure key fields exist
    for needed in ["ad_name", "spend"]:
        if needed not in df2.columns:
            # create empty if missing (we'll still parse what we can)
            df2[needed] = ""

    # Dates
    for dcol in ["dte", "reporting_starts", "reporting_ends"]:
        if dcol in df2.columns:
            df2[dcol] = pd.to_datetime(df2[dcol], errors="coerce").dt.date

    # Numerics / ratios
    if "spend" in df2.columns:
        df2["spend"] = coerce_numeric(df2["spend"]).fillna(0)

    # pick ROAS
    roas = None
    if "roas_purchase_web" in df2:
        roas = coerce_numeric(df2["roas_purchase_web"])
    elif "roas_purchase" in df2:
        roas = coerce_numeric(df2["roas_purchase"])
    if roas is not None:
        df2["roas"] = roas.fillna(0)
    else:
        df2["roas"] = 0.0

    # Cost per purchase (CPR) -> your CPA equivalent
    if "cpr_purchase" in df2:
        df2["cpr_purchase"] = coerce_numeric(df2["cpr_purchase"])
    else:
        df2["cpr_purchase"] = pd.NA

    # Generic CPR from Meta if present
    if "cpr" in df2:
        df2["cpr"] = coerce_numeric(df2["cpr"])
    else:
        df2["cpr"] = pd.NA

    # CPC/CPM/Frequency
    for col in ["cpc", "cpm", "frequency", "avg_play_time"]:
        if col in df2:
            df2[col] = coerce_numeric(df2[col])

    # CTR/hook/hold to ratios
    if "ctr" in df2:
        df2["ctr"] = pct_to_ratio(df2["ctr"]).fillna(0)
    for col in ["hook_rate", "hold_rate"]:
        if col in df2:
            # these often come as ratios already; still coerce
            df2[col] = coerce_numeric(df2[col]).fillna(0)

    # funnel counts
    for col in ["clicks", "atc", "atc_web", "checkout", "checkout_web", "purchases", "purchases_web", "reach"]:
        if col in df2:
            df2[col] = coerce_numeric(df2[col]).fillna(0)

    # choose website-first where available
    if "atc_web" in df2:
        df2["atc_eff"] = df2["atc_web"]
    else:
        df2["atc_eff"] = df2.get("atc", pd.Series(0, index=df2.index))

    if "checkout_web" in df2:
        df2["checkout_eff"] = df2["checkout_web"]
    else:
        df2["checkout_eff"] = df2.get("checkout", pd.Series(0, index=df2.index))

    if "purchases_web" in df2:
        df2["purchases_eff"] = df2["purchases_web"]
    else:
        df2["purchases_eff"] = df2.get("purchases", pd.Series(0, index=df2.index))

    # derive CVR and fallback CPR
    clicks = df2.get("clicks", pd.Series(0, index=df2.index))
    purchases = df2.get("purchases_eff", pd.Series(0, index=df2.index))
    df2["cvr"] = (purchases / clicks.replace(0, pd.NA)).fillna(0.0)

    # If CPR purchase missing, derive spend / purchases
    df2["cpr_eff"] = df2["cpr_purchase"]
    mask_missing = df2["cpr_eff"].isna() & (purchases > 0)
    df2.loc[mask_missing, "cpr_eff"] = (df2.loc[mask_missing, "spend"] / purchases.loc[mask_missing])

    # Fallback to Meta CPR (often cost per result)
    df2["cpr_eff"] = df2["cpr_eff"].fillna(df2["cpr"])

    return df2


def ensure_table_exists():
    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS ad_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ad_name TEXT,
                adset_name TEXT,
                campaign_name TEXT,
                dte DATE,
                reporting_starts DATE,
                reporting_ends DATE,
                delivery_status TEXT,
                delivery_level TEXT,
                spend REAL,
                roas REAL,
                cpr_eff REAL,
                ctr REAL,
                clicks REAL,
                atc_eff REAL,
                checkout_eff REAL,
                purchases_eff REAL,
                cpc REAL,
                cpm REAL,
                reach REAL,
                frequency REAL,
                hook_rate REAL,
                hold_rate REAL,
                avg_play_time REAL
            )
            """)
        )


def append_to_db(df: pd.DataFrame):
    ensure_table_exists()
    cols = [
        "ad_name","adset_name","campaign_name","dte","reporting_starts","reporting_ends",
        "delivery_status","delivery_level","spend","roas","cpr_eff","ctr","clicks","atc_eff",
        "checkout_eff","purchases_eff","cpc","cpm","reach","frequency","hook_rate","hold_rate","avg_play_time"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df[cols].to_sql("ad_metrics", engine, if_exists="append", index=False)


def load_metrics() -> pd.DataFrame:
    ensure_table_exists()
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM ad_metrics"), conn)
    # Coerce types again when reading from DB (SQLite is loose)
    for c in ["spend","roas","cpr_eff","ctr","clicks","atc_eff","checkout_eff","purchases_eff","cpc","cpm","reach","frequency","hook_rate","hold_rate","avg_play_time"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def score_ratio(value: float, target: float) -> float:
    if target is None or target <= 0:
        return 0.0
    if value is None:
        return 0.0
    return max(0.0, min(value / target, 1.0))


def score_inverse(value: float, target: float) -> float:
    # lower is better
    if target is None or target <= 0:
        return 0.0
    if value is None or value <= 0:
        return 1.0  # free pass if zero cost/frequency
    return max(0.0, min(target / value, 1.0))


def growth_score(row: pd.Series, cfg: "AnalyzeConfig") -> float:
    s = 0.0
    s += cfg.w_roas * score_ratio(row.get("roas", 0.0), max(cfg.roas_to_scale, 1e-6))
    s += cfg.w_ctr  * score_ratio(row.get("ctr", 0.0), cfg.target_ctr or 0.015)
    s += cfg.w_hook * score_ratio(row.get("hook_rate", 0.0), cfg.target_hook or 0.30)
    s += cfg.w_hold * score_ratio(row.get("hold_rate", 0.0), cfg.target_hold or 0.25)
    s += cfg.w_cvr  * score_ratio(row.get("cvr", 0.0), cfg.target_cvr or 0.02)
    s += cfg.w_spend* score_ratio(row.get("spend", 0.0), max(cfg.min_spend_for_iterate, 1.0))
    s += cfg.w_cpc  * score_inverse(row.get("cpc", 0.0), cfg.target_cpc or 1.50)
    s += cfg.w_cpm  * score_inverse(row.get("cpm", 0.0), cfg.target_cpm or 25.0)
    s += cfg.w_freq * score_inverse(row.get("frequency", 0.0), cfg.max_frequency or 2.5)
    return round(float(s), 4)


def funnel_diagnostics(row: pd.Series, cfg: "AnalyzeConfig") -> Dict[str, Any]:
    clicks = float(row.get("clicks", 0.0))
    atc    = float(row.get("atc_eff", 0.0))
    chk    = float(row.get("checkout_eff", 0.0))
    pur    = float(row.get("purchases_eff", 0.0))
    ctr    = float(row.get("ctr", 0.0))
    hook   = float(row.get("hook_rate", 0.0))
    hold   = float(row.get("hold_rate", 0.0))

    def safe_rate(num, den): 
        return 0.0 if den <= 0 else num/den

    atc_rate      = safe_rate(atc, clicks)
    checkout_rate = safe_rate(chk, atc)
    purchase_rate = safe_rate(pur, chk)

    issues: List[str] = []

    # Top-of-funnel creative
    if ctr < (cfg.target_ctr or 0.015) * 0.7 or hook < (cfg.target_hook or 0.30) * 0.7:
        issues.append("Creative hook/CTR weak — consider stronger first 2s, clearer thumbstop, and reframe benefits.")
    elif hold < (cfg.target_hold or 0.25) * 0.7:
        issues.append("Audience is stopping early — tighten pacing, pattern breaks around 3–5s, restate promise.")

    # Mid-funnel (LP trust/clarity)
    if atc_rate < 0.10 and clicks >= 50:
        issues.append("Low Click→ATC: landing page likely not conveying value or trust; test offer framing & social proof.")

    # Checkout friction
    if checkout_rate < 0.35 and atc >= 10:
        issues.append("Low ATC→Checkout start: cart friction (shipping surprises, page lag) — streamline steps.")

    # Final intent
    if purchase_rate < 0.4 and chk >= 10:
        issues.append("Low Checkout→Purchase: trust & urgency lacking — add guarantees, testimonials, limited-time nudge.")

    # Frequency fatigue
    if float(row.get("frequency", 0.0)) > (cfg.max_frequency or 2.5) and ctr < (cfg.target_ctr or 0.015):
        issues.append("High frequency with weak CTR — rotate creatives and/or tighten audience.")

    return {
        "ctr": ctr, "hook_rate": hook, "hold_rate": hold,
        "atc_rate": atc_rate, "checkout_rate": checkout_rate, "purchase_rate": purchase_rate,
        "notes": issues
    }


# ---------------------------
# Schemas
# ---------------------------

class AnalyzeConfig(BaseModel):
    # thresholds
    min_spend_for_scale: float = 50.0
    min_spend_for_iterate: float = 20.0
    roas_to_scale: float = 2.0
    target_cpr: Optional[float] = None  # your CPA target (Meta CPR)
    require_cvr: bool = False
    min_cvr: float = 0.0
    # targets for scoring/diagnostics
    target_ctr: float = 0.015
    target_hook: float = 0.30
    target_hold: float = 0.25
    target_cvr: float = 0.02
    target_cpc: float = 1.50
    target_cpm: float = 25.0
    max_frequency: float = 2.5
    # weights for GrowthScore
    w_roas: float = 0.30
    w_ctr: float = 0.12
    w_hook: float = 0.12
    w_hold: float = 0.12
    w_cvr: float = 0.10
    w_spend: float = 0.08
    w_cpc: float = 0.06
    w_cpm: float = 0.05
    w_freq: float = 0.05


class CoachAnswers(BaseModel):
    answers: str


class PromptLabReq(BaseModel):
    target: str = "Meta"
    context: str = ""


class ScriptReq(BaseModel):
    platform: str = "Meta"
    target_seconds: Optional[int] = None
    script: str


# ---------------------------
# UI (no f-string!)
# ---------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0f17; color:#e6edf3; font-family: ui-sans-serif,system-ui,Segoe UI,Roboto,Arial; }
  header { padding:16px 20px; border-bottom:1px solid #1f2633; display:flex; gap:12px; align-items:center; }
  .badge { font-size:12px; padding:4px 8px; background:#1f2633; border-radius:999px; }
  main { padding:20px; max-width:1100px; margin:0 auto; }
  section { border:1px solid #1f2633; border-radius:12px; padding:16px; margin:18px 0; background:#0e1420; }
  h2 { margin:0 0 12px; font-size:18px; }
  input, button, textarea { background:#0b0f17; color:#e6edf3; border:1px solid #263045; padding:8px 10px; border-radius:8px; }
  textarea { width:100%; }
  button { cursor:pointer; }
  pre { background:#0b0f17; border:1px solid #1f2633; padding:12px; border-radius:8px; overflow:auto; white-space:pre-wrap; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .tabs { display:flex; gap:8px; margin-bottom:8px; }
  .tab { padding:6px 10px; border:1px solid #263045; border-radius:8px; cursor:pointer; }
  .active { background:#101a2a; }
  .hide { display:none; }
  label.inline { display:flex; gap:6px; align-items:center; }
</style>
</head>
<body>
<header>
  <strong>__TITLE__</strong>
  <span class="badge">v__VERSION__ • dark</span>
  <a href="/docs" style="margin-left:auto;color:#8fb3ff">OpenAPI Docs →</a>
</header>
<main>
  <div class="tabs">
    <div class="tab active" onclick="show('ingest')">Ingest</div>
    <div class="tab" onclick="show('analyze')">Analyze</div>
    <div class="tab" onclick="show('coach')">Coach</div>
    <div class="tab" onclick="show('prompt')">Prompt Lab</div>
    <div class="tab" onclick="show('script')">Script Analyzer</div>
  </div>

  <section id="ingest">
    <h2>Ingest CSV</h2>
    <div class="row">
      <input type="file" id="csvfile" accept=".csv"/>
      <button onclick="ingest()">Upload</button>
      <button onclick="debugIngest()">Parse-Only</button>
    </div>
    <pre id="ingestOut">Select a CSV and click Upload or Parse-Only.</pre>
  </section>

  <section id="analyze" class="hide">
    <h2>Analyze (GrowthScore + Diagnostics)</h2>
    <div class="row">
      <label>Min Spend Scale <input id="min_spend_scale" type="number" step="0.01" value="50"></label>
      <label>Min Spend Iterate <input id="min_spend_iter" type="number" step="0.01" value="20"></label>
      <label>ROAS to Scale <input id="roas_scale" type="number" step="0.01" value="2.0"></label>
      <label>Target CPR <input id="target_cpr" type="number" step="0.01" placeholder="blank=ignore"></label>
      <label class="inline"><input id="require_cvr" type="checkbox"> Require CVR</label>
      <label>Min CVR <input id="min_cvr" type="number" step="0.001" value="0.0"></label>
    </div>
    <div class="row" style="margin-top:6px">
      <label>Target CTR <input id="t_ctr" type="number" step="0.001" value="0.015"></label>
      <label>Target Hook <input id="t_hook" type="number" step="0.01" value="0.30"></label>
      <label>Target Hold <input id="t_hold" type="number" step="0.01" value="0.25"></label>
      <label>Target CVR <input id="t_cvr" type="number" step="0.001" value="0.02"></label>
      <label>Target CPC <input id="t_cpc" type="number" step="0.01" value="1.50"></label>
      <label>Target CPM <input id="t_cpm" type="number" step="0.01" value="25"></label>
      <label>Max Frequency <input id="t_freq" type="number" step="0.1" value="2.5"></label>
    </div>
    <div class="row" style="margin-top:6px">
      <label>wROAS <input id="w_roas" type="number" step="0.01" value="0.30"></label>
      <label>wCTR <input id="w_ctr" type="number" step="0.01" value="0.12"></label>
      <label>wHook <input id="w_hook" type="number" step="0.01" value="0.12"></label>
      <label>wHold <input id="w_hold" type="number" step="0.01" value="0.12"></label>
      <label>wCVR <input id="w_cvr" type="number" step="0.01" value="0.10"></label>
      <label>wSpend <input id="w_spend" type="number" step="0.01" value="0.08"></label>
      <label>wCPC <input id="w_cpc" type="number" step="0.01" value="0.06"></label>
      <label>wCPM <input id="w_cpm" type="number" step="0.01" value="0.05"></label>
      <label>wFreq <input id="w_freq" type="number" step="0.01" value="0.05"></label>
    </div>
    <div class="row" style="margin-top:10px">
      <button onclick="runAnalyze()">Run Analyze</button>
    </div>
    <pre id="anOut">Click Run Analyze.</pre>
  </section>

  <section id="coach" class="hide">
    <h2>Coach</h2>
    <div class="row"><button onclick="coachQs()">Get Questions</button></div>
    <textarea id="coachAnswers" rows="6" placeholder="Paste your answers here..."></textarea>
    <div class="row" style="margin-top:10px"><button onclick="coachEval()">Evaluate & Objectives</button></div>
    <pre id="coachOut">Use Coach to set objectives.</pre>
  </section>

  <section id="prompt" class="hide">
    <h2>Prompt Lab</h2>
    <div class="row"><input id="pl_target" placeholder="Platform (Meta/TikTok/Google/YouTube)"></div>
    <textarea id="pl_ctx" rows="6" placeholder="Audience, offer, angle, constraints…"></textarea>
    <div class="row" style="margin-top:10px"><button onclick="genPrompt()">Generate Prompt</button></div>
    <pre id="plOut">High-value prompt template will appear here.</pre>
  </section>

  <section id="script" class="hide">
    <h2>Script Analyzer</h2>
    <div class="row">
      <input id="sc_platform" placeholder="Platform (Meta/TikTok/YouTube)"/>
      <input id="sc_len" type="number" step="1" placeholder="Target seconds (e.g., 30)"/>
    </div>
    <textarea id="sc_text" rows="10" placeholder="Paste your script here..."></textarea>
    <div class="row" style="margin-top:10px"><button onclick="analyzeScript()">Analyze Script</button></div>
    <pre id="scOut">Get labeled sections + improvement tips.</pre>
  </section>
</main>

<script>
function show(id){
  ['ingest','analyze','coach','prompt','script'].forEach(s => {
    document.getElementById(s).classList.add('hide');
  });
  document.getElementById(id).classList.remove('hide');

  const tabs = document.querySelectorAll('.tab');
  tabs.forEach(t => t.classList.remove('active'));
  const idx = ['ingest','analyze','coach','prompt','script'].indexOf(id);
  if (idx >= 0) tabs[idx].classList.add('active');
}

async function ingest(){
  const f = document.getElementById('csvfile').files[0];
  if(!f){ alert('Pick a CSV first'); return; }
  const fd = new FormData(); fd.append('file', f, f.name);
  const r = await fetch('/ingest_csv', { method:'POST', body: fd });
  document.getElementById('ingestOut').textContent = await r.text();
}
async function debugIngest(){
  const f = document.getElementById('csvfile').files[0];
  if(!f){ alert('Pick a CSV first'); return; }
  const fd = new FormData(); fd.append('file', f, f.name);
  const r = await fetch('/ingest_csv_debug', { method:'POST', body: fd });
  document.getElementById('ingestOut').textContent = await r.text();
}
async function runAnalyze(){
  const body = {
    min_spend_for_scale: parseFloat(document.getElementById('min_spend_scale').value || '50'),
    min_spend_for_iterate: parseFloat(document.getElementById('min_spend_iter').value || '20'),
    roas_to_scale: parseFloat(document.getElementById('roas_scale').value || '2'),
    target_cpr: document.getElementById('target_cpr').value ? parseFloat(document.getElementById('target_cpr').value) : null,
    require_cvr: document.getElementById('require_cvr').checked,
    min_cvr: parseFloat(document.getElementById('min_cvr').value || '0'),
    target_ctr: parseFloat(document.getElementById('t_ctr').value || '0.015'),
    target_hook: parseFloat(document.getElementById('t_hook').value || '0.30'),
    target_hold: parseFloat(document.getElementById('t_hold').value || '0.25'),
    target_cvr: parseFloat(document.getElementById('t_cvr').value || '0.02'),
    target_cpc: parseFloat(document.getElementById('t_cpc').value || '1.50'),
    target_cpm: parseFloat(document.getElementById('t_cpm').value || '25'),
    max_frequency: parseFloat(document.getElementById('t_freq').value || '2.5'),
    w_roas: parseFloat(document.getElementById('w_roas').value || '0.30'),
    w_ctr: parseFloat(document.getElementById('w_ctr').value || '0.12'),
    w_hook: parseFloat(document.getElementById('w_hook').value || '0.12'),
    w_hold: parseFloat(document.getElementById('w_hold').value || '0.12'),
    w_cvr: parseFloat(document.getElementById('w_cvr').value || '0.10'),
    w_spend: parseFloat(document.getElementById('w_spend').value || '0.08'),
    w_cpc: parseFloat(document.getElementById('w_cpc').value || '0.06'),
    w_cpm: parseFloat(document.getElementById('w_cpm').value || '0.05'),
    w_freq: parseFloat(document.getElementById('w_freq').value || '0.05'),
  };
  const r = await fetch('/analyze', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  document.getElementById('anOut').textContent = await r.text();
}
async function coachQs(){
  const r = await fetch('/coach/questions');
  document.getElementById('coachOut').textContent = await r.text();
}
async function coachEval(){
  const answers = document.getElementById('coachAnswers').value || '';
  const r = await fetch('/coach/evaluate', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ answers })});
  document.getElementById('coachOut').textContent = await r.text();
}
async function genPrompt(){
  const target = document.getElementById('pl_target').value || 'Meta';
  const ctx = document.getElementById('pl_ctx').value || '';
  const r = await fetch('/prompt_lab/generate', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ target, context: ctx })});
  document.getElementById('plOut').textContent = await r.text();
}
async function analyzeScript(){
  const platform = document.getElementById('sc_platform').value || 'Meta';
  const seconds = document.getElementById('sc_len').value ? parseInt(document.getElementById('sc_len').value) : null;
  const script = document.getElementById('sc_text').value || '';
  const r = await fetch('/script/analyze', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ platform, target_seconds: seconds, script })});
  document.getElementById('scOut').textContent = await r.text();
}
</script>
</body>
</html>
"""
    return HTMLResponse(html.replace("__TITLE__", APP_TITLE).replace("__VERSION__", APP_VERSION))

# ---------------------------
# Ingest endpoints
# ---------------------------

@app.post("/ingest_csv")
async def ingest_csv(file: UploadFile = File(...)):
    try:
        df_raw = read_uploaded_csv(file)
        df = normalize_columns(df_raw)
        # If no day column given, try use reporting_starts
        if "dte" not in df.columns or df["dte"].isna().all():
            df["dte"] = df.get("reporting_starts", pd.NaT)

        append_to_db(df)
        return {"status": "ok", "rows_written": int(len(df))}
    except Exception as e:
        set_last_error(e)
        return JSONResponse(
            status_code=500,
            content={"detail": "Server failed to ingest CSV. Hit /debug/last_error for details."}
        )


@app.post("/ingest_csv_debug")
async def ingest_csv_debug(file: UploadFile = File(...)):
    try:
        df_raw = read_uploaded_csv(file)
        df = normalize_columns(df_raw)
        out = {
            "columns": list(df_raw.columns),
            "dtypes": {c: str(df[c].dtype) if c in df else "n/a" for c in df_raw.columns},
            "sample": json.loads(df.head(5).astype(str).to_json(orient="records"))
        }
        return out
    except Exception as e:
        set_last_error(e)
        return JSONResponse(
            status_code=500,
            content={"detail": "Parse failed. Hit /debug/last_error for details."}
        )


@app.get("/debug/last_error")
def last_error():
    return {"last_error": LAST_ERROR or "(empty)"}

# ---------------------------
# Analyze endpoint
# ---------------------------

@app.post("/analyze")
def analyze(cfg: AnalyzeConfig):
    try:
        df = load_metrics()
        if df.empty:
            return {"scale": [], "kill": [], "iterate": [], "potential_winners": [], "diagnostics": [], "creative_gaps": {"needed_new_creatives": 6, "angle_mix": {}, "bans": []}}

        # Group by ad_name over any date range in DB
        g = df.groupby("ad_name", dropna=False).agg({
            "spend":"sum","roas":"mean","cpr_eff":"mean","ctr":"mean","clicks":"sum","atc_eff":"sum","checkout_eff":"sum",
            "purchases_eff":"sum","cpc":"mean","cpm":"mean","reach":"sum","frequency":"mean","hook_rate":"mean","hold_rate":"mean","avg_play_time":"mean"
        }).reset_index()

        # Derived CVR and GrowthScore
        g["cvr"] = (g["purchases_eff"] / g["clicks"].replace(0, pd.NA)).fillna(0.0)
        g["growth_score"] = g.apply(lambda r: growth_score(r, cfg), axis=1)

        # CPR semantics (CPA == CPR on Meta)
        g["cpr"] = g["cpr_eff"]

        scale, kill, iterate, potential = [], [], [], []

        for _, r in g.iterrows():
            ad_id = str(r["ad_name"])
            spend = float(r["spend"])
            roas  = float(r["roas"] or 0)
            cpr   = float(r["cpr"]) if pd.notna(r["cpr"]) else None
            cvr   = float(r["cvr"] or 0)
            gs    = float(r["growth_score"])
            ctr   = float(r["ctr"] or 0)
            hook  = float(r["hook_rate"] or 0)
            hold  = float(r["hold_rate"] or 0)
            freq  = float(r["frequency"] or 0)

            diag = funnel_diagnostics(r, cfg)
            record = {
                "ad_id": ad_id,
                "name": ad_id,
                "spend": round(spend, 2),
                "roas": round(roas, 2),
                "cpr": round(cpr, 2) if cpr is not None else None,
                "cvr": round(cvr, 4),
                "growth_score": gs,
                "ctr": round(ctr, 4),
                "hook_rate": round(hook, 4),
                "hold_rate": round(hold, 4),
                "frequency": round(freq, 2),
                "diagnostics": diag
            }

            meets_cvr = (not cfg.require_cvr) or (cvr >= cfg.min_cvr)
            meets_cpr = (cfg.target_cpr is None) or (cpr is not None and cpr <= cfg.target_cpr)
            meets_roas = roas >= cfg.roas_to_scale

            if spend >= cfg.min_spend_for_scale and meets_cvr and (meets_roas or meets_cpr) and gs >= 0.60:
                record["reason"] = "Hit scale thresholds (ROAS/CPR/CVR) with strong GrowthScore"
                record["probability_to_scale"] = round(gs, 3)
                scale.append(record)
            elif spend >= cfg.min_spend_for_iterate:
                # Kill logic: bad economics or fatigue
                kill_now = False
                reasons = []
                if roas == 0 and spend >= cfg.min_spend_for_iterate:
                    kill_now = True; reasons.append("No purchases at test spend")
                if cfg.target_cpr is not None and cpr is not None and cpr > cfg.target_cpr * 1.3:
                    kill_now = True; reasons.append("CPR well above target")
                if gs < 0.25 and ctr < (cfg.target_ctr or 0.015)*0.6 and hook < (cfg.target_hook or 0.30)*0.6:
                    kill_now = True; reasons.append("Creative underperforming (hook/CTR)")

                if kill_now:
                    record["reason"] = "; ".join(reasons) or "Below breakeven or CPR too high"
                    kill.append(record)
                else:
                    record["reason"] = "Between breakeven and target — iterate"
                    iterate.append(record)
            else:
                # Not enough spend → potential winners if GrowthScore promising
                record["reason"] = "Not enough spend"
                if gs >= 0.50 or (roas >= 1.5) or (cvr >= (cfg.target_cvr or 0.02)*0.7):
                    record["probability_to_scale"] = round(gs, 3)
                    potential.append(record)
                else:
                    iterate.append(record)

        # Creative gaps suggestion (very lightweight heuristic)
        needed_new = max(4, int(len(kill) * 0.5))
        angle_mix = {
            "pain": 40,
            "curiosity": 30,
            "proof": 20,
            "social": 10
        }

        return {
            "scale": scale,
            "kill": kill,
            "iterate": iterate,
            "potential_winners": potential,
            "creative_gaps": {"needed_new_creatives": needed_new, "angle_mix": angle_mix, "bans": []}
        }
    except Exception as e:
        set_last_error(e)
        return JSONResponse(
            status_code=500,
            content={"detail": "Analyze failed. Hit /debug/last_error for details."}
        )

# ---------------------------
# Coach
# ---------------------------

COACH_QUESTIONS = [
    "What is your daily test budget and target CPR (Meta) or CPA?",
    "What audience(s) and placements are you using (any exclusions or stacking)?",
    "What’s the exact offer and price the click goes to?",
    "Do you optimize for Purchase or a proxy (ATC/VC)?",
    "Any strong past winners to clone (angles, hooks, shots)?",
    "What’s your top CRO concern on the LP (speed, trust, clarity)?",
]

@app.get("/coach/questions")
def coach_questions():
    return {"questions": COACH_QUESTIONS}

@app.post("/coach/evaluate")
def coach_evaluate(payload: CoachAnswers):
    text = payload.answers.lower()
    objectives = []
    recs = []

    if "purchase" not in text and "purch" not in text:
        recs.append("Optimize for Purchase event if volume allows; otherwise use ATC with switch once purchases/day > 10.")
    if "exclude" not in text and "exclusion" not in text:
        recs.append("Add exclusions for recent purchasers and heavy engagers to control frequency.")
    if "ugc" in text or "creative" in text:
        objectives.append("Ship 6 new creatives this week across 3 angles (pain, curiosity, proof).")
    else:
        objectives.append("Brief and ship 4 UGC variants focusing on first 2 seconds (pattern breaks).")
    if "speed" in text:
        objectives.append("Raise LP performance to Lighthouse 85+ mobile — compress hero media and inline critical CSS.")
    objectives.append("Set scale rule: ROAS ≥ 2.0 or CPR ≤ target after $50 spend with CVR ≥ 2%.")

    return {"objectives": objectives, "recommendations": recs}

# ---------------------------
# Prompt Lab
# ---------------------------

@app.post("/prompt_lab/generate")
def prompt_lab_generate(req: PromptLabReq):
    t = req.target.strip() or "Meta"
    ctx = req.context.strip()

    template = f"""High-Value {t} Creative/Copy Prompt
Context:
{ctx or "[fill in audience, offer, objections, key benefits]"}

Instructions:
1) Write 5 hooks tailored to the audience pain → benefit in the first 2s.
2) Produce a 25–35s script: Hook (0–3s) → Problem → Agitate → Solution → Proof → Offer → CTA.
3) Include 2 pattern breaks and 1 trust signal (testimonial, guarantee, stat).
4) End with 2 CTAs: primary (purchase) + secondary (learn more).
5) Provide 3 thumbnail headline options.
Output format:
- Hooks (5)
- Script (timestamped beats)
- Thumbnails (3)"""

    return {"prompt": template}

# ---------------------------
# Script Analyzer
# ---------------------------

def split_sentences(text: str) -> List[str]:
    # very simple sentence splitter to avoid extra libs
    parts = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in ".!?":
            s = "".join(buf).strip()
            if s:
                parts.append(s)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts if parts else [text.strip()]

@app.post("/script/analyze")
def script_analyze(req: ScriptReq):
    script = (req.script or "").strip()
    if not script:
        raise HTTPException(status_code=422, detail="Provide a script to analyze.")

    sents = split_sentences(script)

    # naive segmentation
    hook = " ".join(sents[:2]).strip()
    body = sents[2:-1] if len(sents) > 3 else []
    cta  = sents[-1] if len(sents) >= 1 else ""

    problem = ""
    solution = ""
    proof = ""
    offer = ""
    # assign heuristics
    for s in body:
        low = s.lower()
        if any(k in low for k in ["struggle", "tired of", "problem", "frustrat", "hard", "can't"]):
            problem += (" " + s)
        elif any(k in low for k in ["we", "our", "introduc", "here's how", "works"]):
            solution += (" " + s)
        elif any(k in low for k in ["thousand", "5-star", "review", "results", "proof", "before", "after", "guarantee", "warranty"]):
            proof += (" " + s)
        elif any(k in low for k in ["save", "% off", "discount", "today", "now", "limited", "free", "bonus"]):
            offer += (" " + s)

    beats = {
        "hook": hook.strip(),
        "problem": problem.strip() or "(not detected)",
        "solution": solution.strip() or "(not detected)",
        "proof": proof.strip() or "(not detected)",
        "offer": offer.strip() or "(not detected)",
        "cta": cta.strip(),
    }

    # rough timing (if requested)
    timing = None
    if req.target_seconds:
        # weight beats by length
        lengths = {k: len(v) for k, v in beats.items()}
        total = sum(max(1, v) for v in lengths.values())
        timing = {k: round(req.target_seconds * (max(1, lengths[k]) / total)) for k in beats}

    tips = []
    if len(hook.split()) < 8:
        tips.append("Make the hook a concrete payoff promise in ≤ 8 words (and show it visually).")
    if "(not detected)" in beats["problem"]:
        tips.append("State a clear ‘problem’ in the audience’s own words before pitching.")
    if "(not detected)" in beats["proof"]:
        tips.append("Insert 1 strong proof element (testimonial, stat, before/after).")
    if "click" not in beats["cta"].lower() and "buy" not in beats["cta"].lower():
        tips.append("End with a direct CTA (‘Tap Shop Now’ / ‘Get yours today’).")

    return {"beats": beats, "timing": timing, "suggestions": tips, "platform": req.platform}

# ---------------------------
# Small health check
# ---------------------------

@app.get("/healthz")
def healthz():
    return {"ok": True, "version": APP_VERSION}

