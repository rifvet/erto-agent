from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from datetime import date, datetime, timedelta
import pandas as pd
import json
import os
import io

# ======================
# Settings & KPI loading
# ======================
try:
    import settings as cfg  # optional local settings.py
except Exception:
    cfg = object()

def _get(name: str, default: str):
    return getattr(cfg, name, os.getenv(name, default))

DATABASE_URL = _get("DATABASE_URL", "sqlite:///erto_agent.db")

TARGET_ROAS = float(_get("TARGET_ROAS", "2.54"))
BREAKEVEN_ROAS = float(_get("BREAKEVEN_ROAS", "1.54"))
TARGET_CPA = float(_get("TARGET_CPA", "39.30"))
BREAKEVEN_CPA_TRUE_AOV = float(_get("BREAKEVEN_CPA_TRUE_AOV", "81.56"))
BREAKEVEN_CPA_BUY1 = float(_get("BREAKEVEN_CPA_BUY1", "65.50"))  # not used in rules below but kept for future
TARGET_CVR = float(_get("TARGET_CVR", "4.2"))
MIN_SPEND_TO_DECIDE = float(_get("MIN_SPEND_TO_DECIDE", "50"))

BRAND_FILE = "brand_knowledge.json"

# ======================
# App & DB bootstrap
# ======================
app = FastAPI(title="Erto Ad Strategist Agent", version="1.2.0")
engine = create_engine(DATABASE_URL, future=True)

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS ad_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dte TEXT,                 -- ISO date string: YYYY-MM-DD
            ad_id TEXT,
            ad_name TEXT,
            campaign_name TEXT,
            adset_name TEXT,
            spend REAL,
            impressions INTEGER,
            ctr REAL,
            cpc REAL,
            hook_rate REAL,
            hold_rate REAL,
            cvr REAL,
            roas REAL,
            cpa REAL
        );
    """))

# ======================
# Models
# ======================
class AnalyzeRequest(BaseModel):
    # decision & creative plan
    testing_capacity: int = 6
    angle_mix: dict = Field(default_factory=lambda: {"pain": 40, "curiosity": 30, "proof": 20, "social": 10})
    bans: list[str] = Field(default_factory=list)

    # timeframe controls (pick ONE style or leave all empty to use ALL data)
    use_all: bool = False
    days_back: int | None = None  # e.g., 14 for last 14 days (inclusive)
    start_date: str | None = None # "YYYY-MM-DD"
    end_date: str | None = None   # "YYYY-MM-DD"

class DiagnoseRequest(BaseModel):
    site_speed_sec: float | None = None
    # same timeframe controls as analyze
    use_all: bool = False
    days_back: int | None = None
    start_date: str | None = None
    end_date: str | None = None

# ======================
# Helpers
# ======================
NUMERIC_COLS = [
    "spend", "impressions", "ctr", "cpc", "hook_rate", "hold_rate", "cvr", "roas", "cpa",
]

META_RENAME = {
    # Common Meta headers -> internal names
    "Ad ID": "ad_id",
    "Ad Name": "ad_name",
    "Campaign Name": "campaign_name",
    "Ad Set Name": "adset_name",
    "Amount Spent (USD)": "spend",
    "Impressions": "impressions",
    "CTR (All)": "ctr",
    "CPC (Cost per link click)": "cpc",
    # If your CSV already has simple names, leave them; mapping is idempotent
    "spend": "spend",
    "impressions": "impressions",
    "ctr": "ctr",
    "cpc": "cpc",
    "cvr": "cvr",
    "roas": "roas",
    "cpa": "cpa",
}

DATE_CANDIDATES = [
    "dte",
    "Date",
    "date",
    "Day",
    "day",
    "Reporting Starts",
    "Reporting Start",
    "Reporting start",
    "Start Date",
    "start_date",
]

def to_float(val):
    try:
        if val is None or val == "":
            return 0.0
        if isinstance(val, str) and val.strip().endswith("%"):
            return float(val.strip().replace("%", ""))
        return float(val)
    except Exception:
        return 0.0

def load_brand() -> dict:
    try:
        with open(BRAND_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "brand_voice": "Direct, high-signal, no fluff.",
            "positioning": "Quality product with standout durability and comfort.",
            "avatars": [
                {"name": "Main Buyer", "pains": ["overpriced", "low trust"], "motives": ["value", "quality"]}
            ],
            "competitors": ["Generic Brand A", "Brand B"],
            "usps": ["Real materials", "Faster shipping", "Support that cares"],
        }

def resolve_timeframe(use_all: bool, days_back: int | None, start_date: str | None, end_date: str | None):
    """
    Returns a tuple: (mode, params)
      - mode = "all" or "range"
      - params = {} or {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    """
    if use_all:
        return "all", {}

    # explicit range wins if either bound provided
    if start_date or end_date:
        try:
            if start_date:
                sd = datetime.strptime(start_date, "%Y-%m-%d").date()
            else:
                sd = date(1970, 1, 1)
            if end_date:
                ed = datetime.strptime(end_date, "%Y-%m-%d").date()
            else:
                ed = date.today()
            if sd > ed:
                sd, ed = ed, sd  # swap if sent backwards
            return "range", {"start": sd.isoformat(), "end": ed.isoformat()}
        except ValueError:
            # if formatting is wrong, just fall back to all
            return "all", {}

    # days_back if provided
    if days_back is not None and days_back > 0:
        sd = (date.today() - timedelta(days=days_back - 1)).isoformat()
        ed = date.today().isoformat()
        return "range", {"start": sd, "end": ed}

    # default: ALL data user has ingested
    return "all", {}

# ======================
# Routes
# ======================
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ingest_csv")
async def ingest_csv(file: UploadFile = File(...)):
    """
    Upload a Meta CSV. We auto-detect a date column (e.g., Date/Day/Reporting Starts)
    and store it into 'dte' (YYYY-MM-DD). If no date present, we stamp today.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

    # Normalize headers
    df.rename(columns=META_RENAME, inplace=True)

    # Ensure identifiers (support either Ad ID or Ad Name; create both)
    if "ad_id" not in df.columns and "ad_name" in df.columns:
        df["ad_id"] = df["ad_name"].astype(str)
    if "ad_name" not in df.columns and "ad_id" in df.columns:
        df["ad_name"] = df["ad_id"].astype(str)
    df["ad_id"] = df.get("ad_id", "NA").fillna("NA").astype(str)
    df["ad_name"] = df.get("ad_name", df["ad_id"]).fillna("NA").astype(str)

    # Optional text fields
    for col in ["campaign_name", "adset_name"]:
        if col not in df.columns:
            df[col] = ""

    # Add missing metric columns and coerce numerics
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Resolve date column -> dte (ISO)
    found = None
    for cand in DATE_CANDIDATES:
        if cand in df.columns:
            found = cand
            break

    if found:
        dser = pd.to_datetime(df[found], errors="coerce")
        df["dte"] = dser.dt.date.astype(str)
        # If any rows couldn't parse, fill with today
        df["dte"] = df["dte"].where(~df["dte"].isna(), str(date.today()))
    else:
        df["dte"] = str(date.today())

    # Keep only the columns we store
    keep_cols = [
        "dte", "ad_id", "ad_name", "campaign_name", "adset_name",
        "spend", "impressions", "ctr", "cpc", "hook_rate", "hold_rate",
        "cvr", "roas", "cpa",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    try:
        with engine.begin() as conn:
            df.to_sql("ad_metrics", conn, if_exists="append", index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB write error: {e}")

    return {"status": "ok", "rows": int(len(df))}

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    # figure out the timeframe
    mode, params = resolve_timeframe(req.use_all, req.days_back, req.start_date, req.end_date)

    # read rows
    with engine.begin() as conn:
        if mode == "all":
            df = pd.read_sql(text("SELECT * FROM ad_metrics"), conn)
        else:
            df = pd.read_sql(
                text("SELECT * FROM ad_metrics WHERE date(dte) BETWEEN :start AND :end"),
                conn,
                params=params,
            )

    if df.empty:
        return {
            "scale": [],
            "kill": [],
            "iterate": [],
            "creative_gaps": {
                "needed_new_creatives": req.testing_capacity,
                "angle_mix": req.angle_mix,
                "bans": req.bans,
            },
            "window": {"mode": mode, **params},
        }

    # Ensure identifiers
    if "ad_id" not in df.columns:
        df["ad_id"] = "NA"
    if "ad_name" not in df.columns:
        df["ad_name"] = df["ad_id"]

    # Aggregate per-ad across the selected window
    agg = (
        df.groupby("ad_id", as_index=False)
          .agg(
              ad_name=("ad_name", "first"),
              spend=("spend", "sum"),
              impressions=("impressions", "sum"),
              ctr=("ctr", "mean"),
              cpc=("cpc", "mean"),
              hook_rate=("hook_rate", "mean"),
              hold_rate=("hold_rate", "mean"),
              cvr=("cvr", "mean"),
              roas=("roas", "mean"),
              cpa=("cpa", "mean"),
          )
    )

    scale, kill, iterate = [], [], []

    for _, row in agg.iterrows():
        r = row.to_dict()
        label = r.get("ad_name") or r.get("ad_id") or "NA"
        spend = to_float(r.get("spend", 0))
        roas  = to_float(r.get("roas", 0))
        cpa   = to_float(r.get("cpa", 0))
        cvr   = to_float(r.get("cvr", 0))

        item = {
            "ad_id": r.get("ad_id", "NA"),
            "name": label,
            "spend": round(spend, 2),
            "roas": round(roas, 2),
            "cpa": round(cpa, 2),
            "cvr": round(cvr, 2),
        }

        if spend >= MIN_SPEND_TO_DECIDE:
            if roas >= TARGET_ROAS and cpa <= TARGET_CPA and cvr >= TARGET_CVR:
                item["reason"] = "Hit target ROAS/CPA/CVR"
                scale.append(item)
            elif roas < BREAKEVEN_ROAS or cpa > BREAKEVEN_CPA_TRUE_AOV:
                item["reason"] = "Below breakeven or CPA too high"
                kill.append(item)
            else:
                item["reason"] = "Between breakeven and target"
                iterate.append(item)
        else:
            item["reason"] = f"Not enough spend (<{MIN_SPEND_TO_DECIDE})"
            iterate.append(item)

    needed = max(len(kill), req.testing_capacity)
    return {
        "scale": scale,
        "kill": kill,
        "iterate": iterate,
        "creative_gaps": {
            "needed_new_creatives": needed,
            "angle_mix": req.angle_mix,
            "bans": req.bans,
        },
        "window": {"mode": mode, **params},
    }

@app.post("/diagnose")
async def diagnose(req: DiagnoseRequest):
    mode, params = resolve_timeframe(req.use_all, req.days_back, req.start_date, req.end_date)

    with engine.begin() as conn:
        if mode == "all":
            df = pd.read_sql(text("SELECT * FROM ad_metrics"), conn)
        else:
            df = pd.read_sql(
                text("SELECT * FROM ad_metrics WHERE date(dte) BETWEEN :start AND :end"),
                conn,
                params=params,
            )

    if df.empty:
        return {"message": "No data found in the selected window. Upload a CSV via /ingest_csv or widen the window."}

    totals = {
        "spend": float(df.get("spend", pd.Series([0])).sum()),
        "impressions": int(df.get("impressions", pd.Series([0])).sum()),
        "avg_ctr": float(df.get("ctr", pd.Series([0])).mean()),
        "avg_cvr": float(df.get("cvr", pd.Series([0])).mean()),
        "avg_roas": float(df.get("roas", pd.Series([0])).mean()),
        "avg_cpa": float(df.get("cpa", pd.Series([0])).mean()),
    }

    issues = []
    if totals["avg_ctr"] < 1.0:
        issues.append("Low CTR (<1%): hooks/thumbnails not stopping scroll")
    if totals["avg_cvr"] < TARGET_CVR:
        issues.append(f"Low CVR (<{TARGET_CVR}%): offer/landing trust or page flow")
    if totals["avg_roas"] < BREAKEVEN_ROAS:
        issues.append(f"ROAS below breakeven (<{BREAKEVEN_ROAS})")
    if totals["avg_cpa"] > BREAKEVEN_CPA_TRUE_AOV:
        issues.append(f"CPA above breakeven (>{BREAKEVEN_CPA_TRUE_AOV})")

    brand = load_brand()
    prompts = {
        "hooks": (
            "Write 10 scroll-stopping hooks (max 7 words) in the brand voice: "
            f"{brand.get('brand_voice','')}. Target pains: {brand.get('avatars',[{}])[0].get('pains', [])}."
        ),
        "scripts": (
            f"Draft 3x 30s UGC scripts using {brand.get('usps', [])} with curiosity opener → proof → CTA. "
            "Include at least one objection-handle."
        ),
        "lp": "Suggest 5 trust-builders for the product page (microcopy, social proof, risk reversal, badges, reviews block).",
    }

    return {
        "window": {"mode": mode, **params},
        "totals": totals,
        "weak_points": issues or ["No glaring issues vs targets; keep scaling tests running."],
        "recommendations": [
            "Launch 2 new creatives/day: hook-first tests across pain/curiosity/proof/social angles.",
            "If CTR <1%, rework intros & thumbnails. If CVR < target, address risk/reviews/clarity above the fold.",
            "Tighten targeting & bids only after creatives stabilize; avoid early optimization whiplash.",
        ],
        "claude_prompts": prompts,
    }

@app.get("/brand")
async def get_brand():
    return load_brand()

class UpdateBrandRequest(BaseModel):
    brand: dict

@app.post("/update_brand_knowledge")
async def update_brand_knowledge(req: UpdateBrandRequest):
    try:
        with open(BRAND_FILE, "w", encoding="utf-8") as f:
            json.dump(req.brand, f, indent=2, ensure_ascii=False)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
