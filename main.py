from fastapi import FastAPI, UploadFile, File, Body
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from datetime import date
import json, os

from settings import (
    DATABASE_URL,
    TARGET_ROAS, BREAKEVEN_ROAS, TARGET_CPA,
    BREAKEVEN_CPA_TRUE_AOV, TARGET_CVR, EXPERT_MODE
)
from prompts import make_claude_prompt

app = FastAPI(title="ERTO Ad Strategist Agent")
engine = sa.create_engine(DATABASE_URL, future=True)

BRAND_PATH = "brand_knowledge.json"

# Bootstrap tables
with engine.begin() as conn:
    conn.exec_driver_sql("""
    CREATE TABLE IF NOT EXISTS ad_metrics (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      dte TEXT NOT NULL,
      ad_id TEXT NOT NULL,
      spend REAL NOT NULL,
      impressions INTEGER,
      ctr REAL,
      cpc REAL,
      hook_rate REAL,
      hold_rate REAL,
      cvr REAL,
      roas REAL,
      cpa REAL,
      audience TEXT,
      placement TEXT,
      device TEXT
    );
    """)

class AnalyzeRequest(BaseModel):
    testing_capacity: int = 6
    angle_mix: Dict[str,int] = {"pain":40, "curiosity":30, "proof":20, "social":10}
    bans: List[str] = []

class DiagnoseRequest(BaseModel):
    segments: Optional[List[str]] = ["device","placement"]

class BrandUpdate(BaseModel):
    data: Dict[str, Any]

def load_brand():
    if not os.path.exists(BRAND_PATH):
        return {"brand_voice":"", "usps":[], "winning_angles":[], "avatars":[], "positioning":""}
    with open(BRAND_PATH,"r") as f:
        return json.load(f)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/brand")
def get_brand():
    return load_brand()

@app.post("/update_brand_knowledge")
def update_brand(b: BrandUpdate):
    with open(BRAND_PATH, "w") as f:
        json.dump(b.data, f, indent=2)
    return {"status":"updated"}

@app.post("/ingest_csv")
async def ingest_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file).fillna(0) 
    # Normalize common Meta headers → internal names we use everywhere
# Supports both Ad ID and Ad Name. Falls back cleanly if one is missing.
df.rename(columns={
    "Ad ID": "ad_id",
    "Ad Name": "ad_name",
}, inplace=True)

if "ad_id" not in df.columns and "ad_name" in df.columns:
    df["ad_id"] = df["ad_name"].astype(str)
if "ad_name" not in df.columns and "ad_id" in df.columns:
    df["ad_name"] = df["ad_id"].astype(str)

# Make sure both exist (even if CSV lacked both columns)
df["ad_id"] = df.get("ad_id", "NA").fillna("NA").astype(str)
df["ad_name"] = df.get("ad_name", df["ad_id"]).fillna("NA").astype(str)
    if "dte" not in df.columns:
        df["dte"] = str(date.today())
    with engine.begin() as conn:
        for _, r in df.iterrows():
            conn.execute(text("""
                INSERT INTO ad_metrics (dte, ad_id, spend, impressions, ctr, cpc,
                  hook_rate, hold_rate, cvr, roas, cpa, audience, placement, device)
                VALUES (:dte, :ad_id, :spend, :impressions, :ctr, :cpc, :hook_rate,
                  :hold_rate, :cvr, :roas, :cpa, :audience, :placement, :device)
            """), {
                "dte": str(r["dte"]),
                "ad_id": str(r.get("ad_id","NA")),
                "spend": float(r.get("spend",0)),
                "impressions": int(r.get("impressions",0)),
                "ctr": float(r.get("ctr",0)),
                "cpc": float(r.get("cpc",0)),
                "hook_rate": float(r.get("hook_rate",0)),
                "hold_rate": float(r.get("hold_rate",0)),
                "cvr": float(r.get("cvr",0)),
                "roas": float(r.get("roas",0)),
                "cpa": float(r.get("cpa",0)),
                "audience": str(r.get("audience","")),
                "placement": str(r.get("placement","")),
                "device": str(r.get("device","")),
            })
    return {"status":"ok","rows":len(df)}

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

    # Ensure both identifiers are present
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
        spend = float(r.get("spend", 0) or 0)
        roas  = float(r.get("roas", 0) or 0)
        cpa   = float(r.get("cpa", 0) or 0)
        cvr   = float(r.get("cvr", 0) or 0)

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
async def diagnose(_: DiagnoseRequest):
    today = str(date.today())
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM ad_metrics WHERE dte=:d"), conn, params={"d":today})
    if df.empty:
        return {"error":"No data for today"}

    # Derived metrics (simple; replace with real ATC/Checkout columns if present)
    clicks = (df["impressions"]*df["ctr"]/100.0).sum()
    atcs   = (df["impressions"]*df["ctr"]/100.0*df["cvr"]/100.0).sum()
    click_to_atc = (atcs/clicks*100) if clicks>0 else 0

    findings = []
    brand = load_brand()
    if click_to_atc < 4:
        findings.append({
          "diagnosis":"High Clicks, Low ATC",
          "evidence":{"Click→ATC%":round(click_to_atc,2)},
          "interpretation":"Promise–page mismatch and/or trust deficit.",
          "primary_causes_ranked":[
            "Trust layer missing above CTA",
            "Price/returns opacity",
            "Slow LCP or visual clutter"
          ],
          "actions_ordered":[
            "Add trust pack above CTA (stars, 2 micro-reviews, 30-day line, payments/lock)",
            "Show shipping/returns teaser by price; add delivery ETA",
            "Compress hero to ≤200KB WebP; defer non-critical JS; target LCP ≤2.3s"
          ],
          "expected_impact":"+3–5pp Click→ATC within 7 days",
          "verification_plan":"A/B hero+trust vs control; pass if Click→ATC ≥6% and ROAS stable",
          "brand_context":{"voice": brand.get("brand_voice",""), "usps": brand.get("usps",[])}
        })

    # Build Claude prompts from findings + brand knowledge
    prompts = []
    for f in findings:
        if "High Clicks, Low ATC" in f["diagnosis"]:
            prompts.append({
                "angle":"Comfort & Control",
                "audience":"Women 55+ who clicked but didn’t add to cart",
                "prompt": make_claude_prompt(
                    "Comfort & Control",
                    "Women 55+ who clicked but didn’t add to cart",
                    "Trust deficit post-click; reinforce proof and routine-ease",
                    brand.get("brand_voice",""),
                    brand.get("usps",[])
                )
            })

    roas = float(df["roas"].mean())
    cvr  = float(df["cvr"].mean())
    return {"summary":{"clicks":int(clicks),"roas":round(roas,2),"cvr":round(cvr,2),"click_to_atc_pct":round(click_to_atc,2)},
            "findings":findings,"claude_prompts":prompts}

@app.post("/generate_creatives")
def generate_creatives(payload: dict = Body(...)):
    """Given an angle & audience, return a brand-infused Claude prompt."""
    angle = payload.get("angle","Comfort & Control")
    audience = payload.get("audience","Women 45+")
    reason = payload.get("reason","Weekly refresh")
    brand = load_brand()
    return {
        "angle": angle,
        "audience": audience,
        "prompt": make_claude_prompt(
            angle, audience, reason, brand.get("brand_voice",""), brand.get("usps",[])
        )
    }
