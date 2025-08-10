from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from datetime import date
import pandas as pd
import json
import os
import io
import logging

# -------------------------
# Logging & global error cache
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("erto-agent")
LAST_ERROR: str | None = None

# -------------------------
# Settings & KPIs
# -------------------------
try:
    import settings as cfg  # optional settings.py
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

# -------------------------
# App & DB
# -------------------------
app = FastAPI(title="Erto Ad Strategist Agent", version="1.2.0")
engine = create_engine(DATABASE_URL, future=True)

# Create table if not exists
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

# -------------------------
# Models
# -------------------------
class AnalyzeRequest(BaseModel):
    testing_capacity: int = 6
    angle_mix: dict = Field(default_factory=lambda: {"pain": 40, "curiosity": 30, "proof": 20, "social": 10})
    bans: list[str] = Field(default_factory=list)


class DiagnoseRequest(BaseModel):
    site_speed_sec: float | None = None


# -------------------------
# Helpers
# -------------------------
NUMERIC_COLS = [
    "spend", "impressions", "ctr", "cpc", "hook_rate", "hold_rate", "cvr", "roas", "cpa",
]

# Accept a wide set of Meta export header spellings → our internal names
META_RENAME = {
    # IDs / names
    "Ad ID": "ad_id",
    "Ad Name": "ad_name",
    "Ad name": "ad_name",
    "Campaign Name": "campaign_name",
    "Campaign name": "campaign_name",
    "Ad Set Name": "adset_name",
    "Ad set name": "adset_name",

    # Dates
    "Day": "dte",

    # Spend / delivery / reach
    "Amount Spent (USD)": "spend",
    "Amount spent (USD)": "spend",
    "Impressions": "impressions",
    "Reach": "impressions",  # treat reach as impressions if that's what Meta exported

    # Click / rate metrics
    "CTR (All)": "ctr",
    "CTR (link click-through rate)": "ctr",
    "CPC (Cost per link click)": "cpc",
    "CPC (cost per link click)": "cpc",

    # Funnel & revenue-ish
    "Adds to Cart": "add_to_cart",
    "Website adds to cart": "add_to_cart",
    "Initiate Checkout": "initiate_checkout",
    "Checkouts initiated": "initiate_checkout",
    "Website checkouts initiated": "initiate_checkout",
    "Purchases": "purchases",
    "purchases": "purchases",
    "Website purchases": "purchases",
    "Cost per purchase": "cpa",
    "Purchase ROAS (return on ad spend)": "roas",
    "Website purchase ROAS (return on advertising spend)": "roas",

    # Creative metrics
    "hook rate": "hook_rate",
    "hold rate": "hold_rate",
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


def _series_default(value, n):
    """Return a Series of length n filled with value."""
    return pd.Series([value] * n)


def _try_read_csv_from_bytes(raw: bytes) -> pd.DataFrame:
    """Be tolerant to weird encodings/quotes. Return DataFrame or raise."""
    errors = []
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text_data = raw.decode(enc, errors="replace")
            return pd.read_csv(io.StringIO(text_data), engine="python")
        except Exception as e:
            errors.append(f"{enc}: {e}")
    raise ValueError("All read attempts failed: " + " | ".join(errors))


def _coerce_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            # handle percent strings like "3.5%"
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .replace({"": None, "nan": None})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


# -------------------------
# Routes
# -------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest_csv")
async def ingest_csv(file: UploadFile = File(...)):
    """Upload any Meta CSV (any date range). Rows append into ad_metrics."""
    global LAST_ERROR
    try:
        if not file.filename.lower().endswith((".csv")):
            raise HTTPException(status_code=400, detail="Please upload a .csv file")

        raw = await file.read()
        df = _try_read_csv_from_bytes(raw)

        # Normalize headers and drop duplicate columns (keep leftmost)
        df.rename(columns=META_RENAME, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]

        # ---- Ensure identifiers ----
        n = len(df)
        if "ad_id" in df.columns:
            df["ad_id"] = df["ad_id"].astype(str)
        elif "ad_name" in df.columns:
            df["ad_id"] = df["ad_name"].astype(str)
        else:
            df["ad_id"] = _series_default("NA", n)

        if "ad_name" in df.columns:
            df["ad_name"] = df["ad_name"].astype(str)
        else:
            df["ad_name"] = df["ad_id"].astype(str)

        # Optional common fields
        for col in ["campaign_name", "adset_name"]:
            if col not in df.columns:
                df[col] = ""

        # Date column: prefer 'dte' if provided, else 'Day', else today
        if "dte" in df.columns:
            pass
        elif "Day" in df.columns:
            df.rename(columns={"Day": "dte"}, inplace=True)
        else:
            df["dte"] = str(date.today())

        # Create missing numeric metric columns as 0
        for col in NUMERIC_COLS:
            if col not in df.columns:
                df[col] = 0

        # Coerce numerics (handles % strings)
        df = _coerce_numeric_cols(df, NUMERIC_COLS)

        # Final column order
        keep_cols = [
            "dte", "ad_id", "ad_name", "campaign_name", "adset_name",
            "spend", "impressions", "ctr", "cpc", "hook_rate", "hold_rate",
            "cvr", "roas", "cpa",
        ]
        df = df[[c for c in keep_cols if c in df.columns]]

        with engine.begin() as conn:
            df.to_sql("ad_metrics", conn, if_exists="append", index=False)

        return {"status": "ok", "rows": int(len(df)), "columns": list(df.columns)}

    except HTTPException:
        raise
    except Exception as e:
        LAST_ERROR = f"ingest_csv: {type(e).__name__}: {e}"
        logger.exception("ingest_csv failed")
        raise HTTPException(status_code=500, detail="Server failed to ingest CSV. Hit /debug/last_error for details.")


@app.post("/ingest_csv_debug")
async def ingest_csv_debug(file: UploadFile = File(...)):
    """Parse-only: show detected columns, dtypes, and first 5 rows (no DB write)."""
    raw = await file.read()
    df = _try_read_csv_from_bytes(raw)
    df.rename(columns=META_RENAME, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    preview = df.head(5).fillna("").astype(str).to_dict(orient="records")
    dtypes = {k: str(v) for k, v in df.dtypes.items()}
    return {"columns": list(df.columns), "dtypes": dtypes, "sample": preview}


@app.get("/debug/last_error")
async def last_error():
    return {"last_error": LAST_ERROR or "None"}


@app.get("/debug/db_preview")
async def db_preview(limit: int = 20):
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM ad_metrics ORDER BY id DESC LIMIT :lim"), conn, params={"lim": limit})
    return df.fillna("").astype(str).to_dict(orient="records")


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    # analyze ALL rows (no date filter)
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM ad_metrics"), conn)

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

    # Aggregate per ad across all dates
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

    def _to_float(x):
        try:
            if x is None or x == "":
                return 0.0
            if isinstance(x, str) and x.endswith("%"):
                return float(x.replace("%", ""))
            return float(x)
        except Exception:
            return 0.0

    scale, kill, iterate = [], [], []

    for _, row in agg.iterrows():
        r = row.to_dict()
        label = r.get("ad_name") or r.get("ad_id") or "NA"
        spend = _to_float(r.get("spend", 0))
        roas = _to_float(r.get("roas", 0))
        cpa = _to_float(r.get("cpa", 0))
        cvr = _to_float(r.get("cvr", 0))

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
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM ad_metrics"), conn)

    if df.empty:
        return {"message": "No data ingested yet. Upload a CSV via /ingest_csv first."}

    # Totals across all data
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
