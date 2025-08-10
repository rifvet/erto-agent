from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from datetime import date
import pandas as pd
import json
import os
import io

# --- Settings & KPIs ---
try:
    import settings as cfg  # your settings.py
except Exception:
    cfg = object()

def _get(name: str, default: str):
    return getattr(cfg, name, os.getenv(name, default))

DATABASE_URL = _get("DATABASE_URL", "sqlite:///erto_agent.db")
TARGET_ROAS = float(_get("TARGET_ROAS", "2.54"))
BREAKEVEN_ROAS = float(_get("BREAKEVEN_ROAS", "1.54"))
TARGET_CPA = float(_get("TARGET_CPA", "39.30"))
BREAKEVEN_CPA_TRUE_AOV = float(_get("BREAKEVEN_CPA_TRUE_AOV", "81.56"))
BREAKEVEN_CPA_BUY1 = float(_get("BREAKEVEN_CPA_BUY1", "65.50"))
TARGET_CVR = float(_get("TARGET_CVR", "4.2"))
MIN_SPEND_TO_DECIDE = float(_get("MIN_SPEND_TO_DECIDE", "50"))

# --- App & DB ---
app = FastAPI(title="Erto Ad Strategist Agent", version="1.0.0")
engine = create_engine(DATABASE_URL, future=True)

# Create table if it doesn’t exist
with engine.begin() as conn:
    conn.execute(text(
        """
        CREATE TABLE IF NOT EXISTS ad_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dte TEXT,
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
        """
    ))

# --- Models ---
class AnalyzeRequest(BaseModel):
    testing_capacity: int = 6
    angle_mix: dict = Field(default_factory=lambda: {"pain": 40, "curiosity": 30, "proof": 20, "social": 10})
    bans: list[str] = Field(default_factory=list)

class DiagnoseRequest(BaseModel):
    site_speed_sec: float | None = None

# --- Helpers ---
NUMERIC_COLS = [
    "spend", "impressions", "ctr", "cpc", "hook_rate", "hold_rate", "cvr", "roas", "cpa",
]

META_RENAME = {
    # Common Meta export headers → internal
    "Ad ID": "ad_id",
    "Ad Name": "ad_name",
    "Campaign Name": "campaign_name",
    "Ad Set Name": "adset_name",
    "Amount Spent (USD)": "spend",
    "Impressions": "impressions",
    "CTR (All)": "ctr",
    "CPC (Cost per link click)": "cpc",
    "Adds to Cart": "add_to_cart",
    "Initiate Checkout": "initiate_checkout",
    "Purchases": "purchases",
    # Sometimes exports already come simplified; mapping won’t hurt
    "spend": "spend",
    "impressions": "impressions",
    "ctr": "ctr",
    "cpc": "cpc",
    "cvr": "cvr",
    "roas": "roas",
    "cpa": "cpa",
}

BRAND_FILE = "brand_knowledge.json"


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


def to_float(val):
    try:
        if val is None or val == "":
            return 0.0
        if isinstance(val, str) and val.endswith("%"):
            return float(val.replace("%", ""))
        return float(val)
    except Exception:
        return 0.0

# --- Routes ---
@app.get("/health")
async def health():
    return {"status": "ok"}


from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import io
import pandas as pd
from sqlalchemy import text
from datetime import date

from fastapi import Query

@app.post("/ingest_csv")
async def ingest_csv(
    file: UploadFile = File(...),
    dry_run: bool = Query(False, description="Parse only; don't write to DB")
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    # 1) Read CSV
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

    # 2) Normalize headers you commonly get from Meta
    df.rename(columns=META_RENAME, inplace=True)

    # Ensure identifiers exist
    if "ad_id" not in df.columns and "ad_name" in df.columns:
        df["ad_id"] = df["ad_name"].astype(str)
    if "ad_name" not in df.columns and "ad_id" in df.columns:
        df["ad_name"] = df["ad_id"].astype(str)
    df["ad_id"] = df.get("ad_id", "NA").fillna("NA").astype(str)
    df["ad_name"] = df.get("ad_name", df["ad_id"]).fillna("NA").astype(str)

    # Optional common fields
    for col in ["campaign_name", "adset_name"]:
        if col not in df.columns:
            df[col] = ""

    # Fill/convert numeric metrics
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Add date
    if "dte" not in df.columns:
        df["dte"] = str(date.today())

    keep_cols = [
        "dte", "ad_id", "ad_name", "campaign_name", "adset_name",
        "spend", "impressions", "ctr", "cpc", "hook_rate", "hold_rate",
        "cvr", "roas", "cpa",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    # 3) Dry-run shows us what we parsed (no DB write)
    if dry_run:
        return {
            "status": "dry_run_ok",
            "rows": int(len(df)),
            "columns": list(df.columns),
            "preview": df.head(5).to_dict(orient="records"),
        }

    # 4) Write to DB with explicit error surfacing
    try:
        with engine.begin() as conn:
            df.to_sql("ad_metrics", conn, if_exists="append", index=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DB write failed: {e}")

    return {"status": "ok", "rows": int(len(df))}



@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    today = str(date.today())
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM ad_metrics WHERE dte=:d"), conn, params={"d": today})

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
        }

    # Ensure identifiers
    if "ad_id" not in df.columns:
        df["ad_id"] = "NA"
    if "ad_name" not in df.columns:
        df["ad_name"] = df["ad_id"]

    # Aggregate per ad
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
    }


@app.post("/diagnose")
async def diagnose(req: DiagnoseRequest):
    today = str(date.today())
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM ad_metrics WHERE dte=:d"), conn, params={"d": today})

    if df.empty:
        return {"message": "No data ingested today. Upload a CSV via /ingest_csv first."}

    # Totals (use available columns)
    totals = {
        "spend": float(df.get("spend", pd.Series([0])).sum()),
        "impressions": int(df.get("impressions", pd.Series([0])).sum()),
        "avg_ctr": float(df.get("ctr", pd.Series([0])).mean()),
        "avg_cvr": float(df.get("cvr", pd.Series([0])).mean()),
        "avg_roas": float(df.get("roas", pd.Series([0])).mean()),
        "avg_cpa": float(df.get("cpa", pd.Series([0])).mean()),
    }

    # Simple weak-point logic
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

    # Claude-ready prompts (you paste into Claude)
    prompts = {
        "hooks": f"Write 10 scroll-stopping hooks (max 7 words) in the brand voice: {brand.get('brand_voice','')}. Target pains: {brand.get('avatars',[{}])[0].get('pains', [])}.",
        "scripts": f"Draft 3x 30s UGC scripts using {brand.get('usps', [])} with curiosity opener → proof → CTA. Include at least one objection-handle.",
        "lp": "Suggest 5 trust-builders for the product page (microcopy, social proof, risk reversal, badges, reviews block).",
    }

    return {
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

# (No need for if __name__ == "__main__": as Render runs uvicorn directly)
