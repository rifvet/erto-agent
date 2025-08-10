import os
import io
import json
import math
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

APP_TITLE = "ERTO Agent"
APP_VERSION = "1.8.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Globals
# ---------------------------
_last_error: Dict[str, Any] = {"error": None}
def set_last_error(msg: str, detail: Any = None):
    global _last_error
    _last_error = {"error": msg, "detail": detail, "ts": datetime.utcnow().isoformat()+"Z"}

# ---------------------------
# DB (SQLite fallback)
# ---------------------------
def get_engine() -> Engine:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        url = "sqlite:///./ad_metrics.db"
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    return create_engine(url, future=True)

ENGINE = get_engine()
with ENGINE.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS ad_metrics (
            date TEXT,
            campaign TEXT,
            adset TEXT,
            ad_name TEXT,
            spend REAL,
            purchases REAL,
            roas REAL,
            cpr REAL,
            clicks REAL,
            reach REAL,
            ctr REAL,
            hook_rate REAL,
            hold_rate REAL,
            cpc REAL,
            cpm REAL,
            frequency REAL,
            atc REAL,
            checkout REAL,
            avg_play_time REAL,
            raw JSON
        )
    """))

# ---------------------------
# CSV utils
# ---------------------------
def read_csv_upload(file: UploadFile) -> pd.DataFrame:
    try:
        content = file.file.read()
        return pd.read_csv(io.BytesIO(content), encoding_errors="replace", engine="python")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # map MANY possible Meta export headers -> our canonical names
    mapping = {
        "ad name": "ad_name", "ad": "ad_name",
        "ad set name": "adset", "adset name": "adset", "adset": "adset",
        "campaign name": "campaign", "campaign": "campaign",
        "day": "date", "reporting starts": "date_start", "reporting ends": "date_end",

        "amount spent (usd)": "spend", "spend": "spend",

        "website purchases": "purchases", "purchases": "purchases",

        # ROAS / CPR / CPA
        "purchase roas (return on ad spend)": "roas",
        "website purchase roas (return on advertising spend)": "roas",
        "roas": "roas",
        "cost per purchase": "cpr",
        "cost per result": "cpr",  # treat CPA as CPR on Meta like you asked

        # Clicks / CTR
        "link clicks": "clicks", "clicks": "clicks",
        "ctr (link click-through rate)": "ctr", "ctr": "ctr",
        "cpc (cost per link click)": "cpc", "cpc": "cpc",

        # CPM / Frequency
        "cpm (cost per 1,000 impressions)": "cpm", "cpm": "cpm",
        "frequency": "frequency",

        # Creative quality
        "hook rate": "hook_rate", "hold rate": "hold_rate",
        "video average play time": "avg_play_time",

        # Funnel steps
        "adds to cart": "atc", "website adds to cart": "atc",
        "checkouts initiated": "checkout", "website checkouts initiated": "checkout",

        # misc
        "reach": "reach",
        "result type": "result_type",
        "delivery status": "delivery_status",
        "delivery level": "delivery_level",
    }

    renames = {}
    for c in df.columns:
        key = c.strip().lower()
        if key in mapping:
            renames[c] = mapping[key]
    df = df.rename(columns=renames)

    # Ensure date
    if "date" not in df.columns:
        if "date_start" in df.columns: df["date"] = df["date_start"]
        elif "Day" in df.columns: df["date"] = df["Day"]
        elif "Reporting starts" in df.columns: df["date"] = df["Reporting starts"]

    # Numerics
    numeric = ["spend","purchases","roas","cpr","clicks","reach","ctr","hook_rate","hold_rate",
               "cpc","cpm","frequency","atc","checkout","avg_play_time"]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # CTR sometimes exported as %
    if "ctr" in df.columns and (df["ctr"].dropna() > 1).any():
        df["ctr"] = df["ctr"]/100.0

    # Date parse
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Defaults
    for c in ["spend","purchases","clicks","reach","roas","cpr","hook_rate","hold_rate","atc","checkout"]:
        if c in df.columns: df[c] = df[c].fillna(0)

    for c in ["ad_name","adset","campaign"]:
        if c in df.columns: df[c] = df[c].astype(str)

    return df

def safe_div(a: float, b: float) -> float:
    try:
        a = float(a or 0)
        b = float(b or 0)
    except Exception:
        return 0.0
    if b <= 0: return 0.0
    return a / b

def compute_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # CPR fallback if missing but we have spend & purchases
    if "cpr" not in df.columns:
        df["cpr"] = 0.0
    if "purchases" in df.columns and "spend" in df.columns:
        mask = (df["cpr"].fillna(0) == 0) & (df["purchases"].fillna(0) > 0)
        df.loc[mask, "cpr"] = df.loc[mask, "spend"] / df.loc[mask, "purchases"]

    # Stage rates at row-level (will re-compute after aggregation as well)
    df["cvr"] = df.apply(lambda r: safe_div(r.get("purchases",0), r.get("clicks",0)), axis=1)  # Purchases / Clicks
    df["atc_rate"] = df.apply(lambda r: safe_div(r.get("atc",0), r.get("clicks",0)), axis=1)
    df["checkout_rate"] = df.apply(lambda r: safe_div(r.get("checkout",0), r.get("atc",0)), axis=1)
    df["purchase_rate"] = df.apply(lambda r: safe_div(r.get("purchases",0), r.get("checkout",0)), axis=1)
    return df

def aggregate_by_ad(df: pd.DataFrame) -> pd.DataFrame:
    if "ad_name" not in df.columns:
        df["ad_name"] = df.get("adset", df.get("campaign", "UNKNOWN"))
    agg = {
        "spend":"sum","purchases":"sum","clicks":"sum","reach":"sum","atc":"sum","checkout":"sum",
        "roas":"mean","cpr":"mean","ctr":"mean","hook_rate":"mean","hold_rate":"mean",
        "cpc":"mean","cpm":"mean","frequency":"mean","avg_play_time":"mean",
    }
    agg = {k:v for k,v in agg.items() if k in df.columns}
    g = df.groupby(["ad_name"], dropna=False).agg(agg).reset_index()

    # Stage rates after aggregation
    g["cvr"] = g.apply(lambda r: safe_div(r.get("purchases",0), r.get("clicks",0)), axis=1)
    g["atc_rate"] = g.apply(lambda r: safe_div(r.get("atc",0), r.get("clicks",0)), axis=1)
    g["checkout_rate"] = g.apply(lambda r: safe_div(r.get("checkout",0), r.get("atc",0)), axis=1)
    g["purchase_rate"] = g.apply(lambda r: safe_div(r.get("purchases",0), r.get("checkout",0)), axis=1)
    return g

def safe_to_sql(df: pd.DataFrame, table="ad_metrics"):
    raw_cols = list(df.columns)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "date": str(r.get("date")) if not pd.isna(r.get("date")) else None,
            "campaign": r.get("campaign"),
            "adset": r.get("adset"),
            "ad_name": r.get("ad_name"),
            "spend": float(r.get("spend",0) or 0),
            "purchases": float(r.get("purchases",0) or 0),
            "roas": float(r.get("roas",0) or 0),
            "cpr": float(r.get("cpr",0) or 0),
            "clicks": float(r.get("clicks",0) or 0),
            "reach": float(r.get("reach",0) or 0),
            "ctr": float(r.get("ctr",0) or 0),
            "hook_rate": float(r.get("hook_rate",0) or 0),
            "hold_rate": float(r.get("hold_rate",0) or 0),
            "cpc": float(r.get("cpc",0) or 0),
            "cpm": float(r.get("cpm",0) or 0),
            "frequency": float(r.get("frequency",0) or 0),
            "atc": float(r.get("atc",0) or 0),
            "checkout": float(r.get("checkout",0) or 0),
            "avg_play_time": float(r.get("avg_play_time",0) or 0),
            "raw": json.dumps({c:(None if pd.isna(r.get(c)) else r.get(c)) for c in raw_cols}),
        })
    if rows:
        pd.DataFrame(rows).to_sql(table, ENGINE, if_exists="append", index=False)

# ---------------------------
# Schemas
# ---------------------------
class AnalyzeParams(BaseModel):
    # decision thresholds
    min_spend_for_scale: float = 50.0
    min_spend_for_iterate: float = 20.0
    roas_to_scale: float = 2.0
    kill_if_roas_below: float = 0.9

    # CPR target (Meta). `target_cpa` kept for backward-compat.
    target_cpr: Optional[float] = None
    target_cpa: Optional[float] = None  # alias

    require_cvr: bool = False
    min_cvr: float = 0.0
    lookback_days: Optional[int] = None

    # growth-score targets
    target_roas: float = 2.0
    target_ctr: float = 0.015
    target_hook: float = 0.30
    target_hold: float = 0.25
    target_cvr: float = 0.02
    target_cpc: float = 1.50
    target_cpm: float = 25.0
    max_frequency: float = 2.5

    # growth-score weights (don’t need to sum to 1; we cap)
    w_roas: float = 0.30
    w_ctr: float = 0.12
    w_hook: float = 0.12
    w_hold: float = 0.12
    w_cvr: float = 0.10
    w_spend: float = 0.08
    w_cpc: float = 0.06
    w_cpm: float = 0.05
    w_freq: float = 0.05

    # potential winners band
    potential_min_spend: float = 10.0
    potential_max_spend: float = 50.0
    potential_score_threshold: float = 55.0  # 0-100

class ActionItem(BaseModel):
    ad_id: str
    name: str
    spend: float
    roas: float
    cpr: float
    cvr: float
    ctr: float = 0.0
    hook_rate: float = 0.0
    hold_rate: float = 0.0
    atc_rate: float = 0.0
    checkout_rate: float = 0.0
    purchase_rate: float = 0.0
    probability_to_grow: float = 0.0  # 0-100
    reason: str

class CreativeGaps(BaseModel):
    needed_new_creatives: int
    angle_mix: Dict[str, int]
    bans: List[str] = []

class FunnelReport(BaseModel):
    stage_rates: Dict[str, float]
    biggest_bottleneck: str
    probable_causes: List[str]
    recommended_experiments: List[str]

class PerAdIssue(BaseModel):
    ad_id: str
    name: str
    stage_bottleneck: str
    evidence: Dict[str, float]
    hypothesis: str
    experiments: List[str]

class AnalyzeResponse(BaseModel):
    scale: List[ActionItem]
    kill: List[ActionItem]
    iterate: List[ActionItem]
    potential_winners: List[ActionItem]
    creative_gaps: CreativeGaps
    top_by_growth: List[ActionItem]
    account_diagnostics: FunnelReport
    per_ad_issues: List[PerAdIssue]

# ---------------------------
# Root (dark UI) + Debug
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{APP_TITLE}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#0b0f17; color:#e6edf3; font-family: ui-sans-serif,system-ui,Segoe UI,Roboto,Arial; }}
  header {{ padding:16px 20px; border-bottom:1px solid #1f2633; display:flex; gap:12px; align-items:center; }}
  .badge {{ font-size:12px; padding:4px 8px; background:#1f2633; border-radius:999px; }}
  main {{ padding:20px; max-width:1100px; margin:0 auto; }}
  section {{ border:1px solid #1f2633; border-radius:12px; padding:16px; margin:18px 0; background:#0e1420; }}
  h2 {{ margin:0 0 12px; font-size:18px; }}
  input, button, textarea {{ background:#0b0f17; color:#e6edf3; border:1px solid #263045; padding:8px 10px; border-radius:8px; }}
  textarea {{ width:100%; }}
  button {{ cursor:pointer; }}
  pre {{ background:#0b0f17; border:1px solid #1f2633; padding:12px; border-radius:8px; overflow:auto; }}
  .row {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
  .tabs {{ display:flex; gap:8px; margin-bottom:8px; }}
  .tab {{ padding:6px 10px; border:1px solid #263045; border-radius:8px; cursor:pointer; }}
  .active {{ background:#101a2a; }}
  .hide {{ display:none; }}
  label.inline {{ display:flex; gap:6px; align-items:center; }}
</style>
</head>
<body>
<header>
  <strong>ERTO Agent</strong>
  <span class="badge">v{APP_VERSION} • dark</span>
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
  for (const s of ['ingest','analyze','coach','prompt','script']){
    document.getElementById(s).classList.add('hide');
  }
  document.getElementById(id).classList.remove('hide');
  for (const el of document.querySelectorAll('.tab')) el.classList.remove('active');
  const idx = ['ingest','analyze','coach','prompt','script'].indexOf(id);
  document.querySelectorAll('.tab')[idx].classList.add('active');
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

@app.get("/debug/last_error")
def last_error():
    return JSONResponse(_last_error)

# ---------------------------
# Ingest
# ---------------------------
@app.post("/ingest_csv_debug")
async def ingest_csv_debug(file: UploadFile = File(...)):
    try:
        df = read_csv_upload(file)
        df = normalize_columns(df)
        df = compute_row_metrics(df)
        sample = df.head(5).fillna("").astype(str).to_dict(orient="records")
        dtypes = {c: str(df[c].dtype) for c in df.columns}
        return JSONResponse({"columns": list(df.columns), "dtypes": dtypes, "sample": sample})
    except Exception as e:
        set_last_error("ingest_csv_debug failed", str(e)); raise

@app.post("/ingest_csv")
async def ingest_csv(file: UploadFile = File(...)):
    try:
        df = read_csv_upload(file)
        df = normalize_columns(df)
        df = compute_row_metrics(df)
        safe_to_sql(df)
        return PlainTextResponse(f"OK: {len(df)} rows ingested into ad_metrics")
    except Exception as e:
        set_last_error("ingest_csv failed", str(e))
        raise HTTPException(status_code=500, detail="Server failed to ingest CSV. Hit /debug/last_error for details.")

# ---------------------------
# Analyze helpers
# ---------------------------
def _norm_higher_better(x: float, target: float, cap_multiple: float = 2.0) -> float:
    if target <= 0: return 0.0
    r = max(0.0, min(x/target, cap_multiple))
    return r / cap_multiple

def _norm_lower_better(x: float, target: float) -> float:
    if target <= 0: return 0.0
    # 1.0 at or below target; decays as it gets worse
    r = target / max(x, target)
    return max(0.0, min(r, 1.0))

def _spend_factor(spend: float, min_spend_for_scale: float) -> float:
    if min_spend_for_scale <= 0: return 0.0
    return max(0.0, min(1.0, spend / min_spend_for_scale))

def growth_score(row: pd.Series, p: AnalyzeParams) -> float:
    roas = float(row.get("roas",0) or 0)
    ctr  = float(row.get("ctr",0) or 0)
    hook = float(row.get("hook_rate",0) or 0)
    hold = float(row.get("hold_rate",0) or 0)
    cvr  = float(row.get("cvr",0) or 0)
    spend= float(row.get("spend",0) or 0)
    cpc  = float(row.get("cpc",0) or 0)
    cpm  = float(row.get("cpm",0) or 0)
    freq = float(row.get("frequency",0) or 0)

    score = (
        p.w_roas * _norm_higher_better(roas, p.target_roas) +
        p.w_ctr  * _norm_higher_better(ctr,  p.target_ctr)  +
        p.w_hook * _norm_higher_better(hook, p.target_hook) +
        p.w_hold * _norm_higher_better(hold, p.target_hold) +
        p.w_cvr  * _norm_higher_better(cvr,  p.target_cvr)  +
        p.w_spend* _spend_factor(spend, p.min_spend_for_scale) +
        p.w_cpc  * _norm_lower_better(cpc if cpc>0 else p.target_cpc, p.target_cpc) +
        p.w_cpm  * _norm_lower_better(cpm if cpm>0 else p.target_cpm, p.target_cpm) +
        p.w_freq * (1.0 if freq==0 else _norm_lower_better(freq, p.max_frequency))
    )
    return round(100.0 * max(0.0, min(score, 1.0)), 2)

def mk_item(row: pd.Series, reason: str, prob: float) -> ActionItem:
    return ActionItem(
        ad_id=str(row["ad_name"]),
        name=str(row["ad_name"]),
        spend=float(row.get("spend",0) or 0),
        roas=float(row.get("roas",0) or 0),
        cpr=float(row.get("cpr",0) or 0),
        cvr=float(row.get("cvr",0) or 0),
        ctr=float(row.get("ctr",0) or 0),
        hook_rate=float(row.get("hook_rate",0) or 0),
        hold_rate=float(row.get("hold_rate",0) or 0),
        atc_rate=float(row.get("atc_rate",0) or 0),
        checkout_rate=float(row.get("checkout_rate",0) or 0),
        purchase_rate=float(row.get("purchase_rate",0) or 0),
        probability_to_grow=float(prob),
        reason=reason
    )

def account_funnel_report(g: pd.DataFrame) -> FunnelReport:
    # weighted by clicks/atc/checkout appropriately
    total_clicks = g["clicks"].sum() if "clicks" in g.columns else 0
    total_atc = g["atc"].sum() if "atc" in g.columns else 0
    total_checkout = g["checkout"].sum() if "checkout" in g.columns else 0
    total_pur = g["purchases"].sum() if "purchases" in g.columns else 0

    ctr = g["ctr"].mean() if "ctr" in g.columns and not g["ctr"].isna().all() else 0.0
    atc_rate = safe_div(total_atc, total_clicks)
    checkout_rate = safe_div(total_checkout, total_atc)
    purchase_rate = safe_div(total_pur, total_checkout)
    cvr = safe_div(total_pur, total_clicks)

    # Find biggest drop
    stages = {
        "Awareness→Click (CTR)": ctr,
        "Click→ATC": atc_rate,
        "ATC→Checkout": checkout_rate,
        "Checkout→Purchase": purchase_rate
    }
    # “Bottleneck” = the smallest rate among the post-click stages; CTR poor is creative/audience
    post_click = {k:v for k,v in stages.items() if "CTR" not in k}
    bottleneck = min(post_click, key=post_click.get) if post_click else "Click→ATC"

    causes, experiments = [], []
    # Heuristics for causes/experiments
    if bottleneck == "Click→ATC":
        causes = [
            "Landing page-message mismatch with ad promise",
            "Slow load / poor mobile UX",
            "Weak first-screen trust (reviews, badges, social proof)",
            "Price anchoring or variant confusion"
        ]
        experiments = [
            "Match headline to ad hook; repeat payoff above the fold",
            "Move social proof (stars, UGC quotes) into first viewport",
            "Speed test + compress hero; remove render-blocking scripts",
            "Add price anchor/compare-at; simplify options; default best-seller"
        ]
    elif bottleneck == "ATC→Checkout":
        causes = [
            "Cart UX friction (popups, surprise fees)",
            "Shipping/returns unclear, trust not reassured",
            "Upsell clutter before checkout"
        ]
        experiments = [
            "One-click cart → checkout; delay upsells post-purchase",
            "Show shipping/returns/promise badges in cart",
            "Cart reminder micro-copy: delivery time, guarantee"
        ]
    elif bottleneck == "Checkout→Purchase":
        causes = [
            "Checkout friction (fields, account creation)",
            "Payment method gaps; address validation issues",
            "Last-minute price shock (tax, shipping)"
        ]
        experiments = [
            "Enable guest checkout + autofill; reduce required fields",
            "Add ShopPay/PayPal/Apple Pay; test address autocomplete",
            "Inline total cost disclosure; free shipping threshold test"
        ]

    if ctr < 0.012:
        causes.insert(0, "Creative not compelling or wrong audience")
        experiments.insert(0, "New hooks/angles; refresh openers; test different audience/geo")

    return FunnelReport(
        stage_rates={"ctr": round(ctr,4), "atc_rate": round(atc_rate,4), "checkout_rate": round(checkout_rate,4),
                     "purchase_rate": round(purchase_rate,4), "cvr": round(cvr,4)},
        biggest_bottleneck=bottleneck,
        probable_causes=causes[:6],
        recommended_experiments=experiments[:6]
    )

def per_ad_issues(g: pd.DataFrame) -> List[PerAdIssue]:
    issues: List[PerAdIssue] = []
    for _, r in g.iterrows():
        ctr = float(r.get("ctr",0) or 0)
        atc_rate = float(r.get("atc_rate",0) or 0)
        checkout_rate = float(r.get("checkout_rate",0) or 0)
        purchase_rate = float(r.get("purchase_rate",0) or 0)

        # simple rules to call out biggest weak link per ad
        scores = {
            "Click→ATC": atc_rate,
            "ATC→Checkout": checkout_rate,
            "Checkout→Purchase": purchase_rate
        }
        stage = min(scores, key=scores.get)
        hyp = "Landing page mismatch" if stage=="Click→ATC" else ("Checkout friction" if stage=="Checkout→Purchase" else "Cart friction / trust gap")

        exps = {
            "Click→ATC": [
                "Rewrite hero to mirror ad hook",
                "Add review stars & UGC quote above the fold",
                "Shorten hero copy; tighter bullets"
            ],
            "ATC→Checkout": [
                "Remove pre-checkout upsell; highlight free returns",
                "Simplify cart; persistent checkout button above the fold"
            ],
            "Checkout→Purchase": [
                "Enable guest checkout; reduce fields",
                "Add ShopPay/PayPal; reveal total cost early"
            ],
        }
        issues.append(PerAdIssue(
            ad_id=str(r["ad_name"]), name=str(r["ad_name"]), stage_bottleneck=stage,
            evidence={
                "ctr": round(ctr,4), "atc_rate": round(atc_rate,4),
                "checkout_rate": round(checkout_rate,4), "purchase_rate": round(purchase_rate,4)
            },
            hypothesis=hyp,
            experiments=exps[stage]
        ))
    return issues

# ---------------------------
# Analyze
# ---------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(params: AnalyzeParams):
    try:
        # CPR aliasing
        if params.target_cpr is None and params.target_cpa is not None:
            params.target_cpr = params.target_cpa

        where = []
        if params.lookback_days:
            cutoff = (datetime.utcnow().date() - pd.Timedelta(days=params.lookback_days)).isoformat()
            where.append(f"date >= '{cutoff}'")
        sql = "SELECT * FROM ad_metrics" + ((" WHERE " + " AND ".join(where)) if where else "")
        df = pd.read_sql(sql, ENGINE)
        if df.empty:
            empty_fr = FunnelReport(stage_rates={"ctr":0,"atc_rate":0,"checkout_rate":0,"purchase_rate":0,"cvr"_
