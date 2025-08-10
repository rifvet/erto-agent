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
# ERTO Agent — lightweight, DB-free version
# FastAPI app with: CSV ingest, Coach, Prompt Lab, Script Doctor, and a clean dark UI.

import io
import os
import json
import math
from typing import Optional, List, Dict, Any

import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

APP_NAME = "Erto Media Agent"
import pandas as pd
import numpy as np
import io
import traceback
import datetime as dt

app = FastAPI(title=APP_NAME)
app = FastAPI(title="ERTO Agent", version="1.0.0")

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
# --------------------------------------------------------------------------------------
# In-memory store (keeps it simple & avoids DB 500s). You can swap to a DB later.
# --------------------------------------------------------------------------------------
LAST_DF: pd.DataFrame | None = None
LAST_ERROR: str | None = None

# Defaults / thresholds
SETTINGS = {
    "breakeven_roas": 1.54,
    "target_roas": 2.0,
    "min_test_spend": 20.0,
    "min_scale_spend": 50.0,
    "hook_good": 0.30,   # 30%
    "hold_good": 0.10,   # 10%
    "ctr_good": 0.015,   # 1.5%
    "assume_click_attrib_ok": False,   # set True if you know 7d-click >= 60%
    "external_data": False,            # placeholder for future external intelligence
}

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def set_last_error(e: Exception):
    global LAST_ERROR
    LAST_ERROR = "".join(traceback.format_exception(type(e), e, e.__traceback__))

def _to_num(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if pd.isna(x):
            return None
        return float(x)
def _to_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    try:
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        val = float(s)
        return val
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
        return np.nan

def _pct_to_decimal(x):
    """If a 'percentage-looking' number is >1, assume it was given as 3.5 meaning 3.5%."""
    v = _to_float(x)
    if pd.isna(v):
        return np.nan
    return v / 100.0 if v > 1.0 else v

def _first_existing(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map many Meta export variants into a normalized schema."""
    df = df.copy()

    # Standard name mapping guesses
    col_ad        = _first_existing(df, ["Ad name", "Ad", "Ad Name"])
    col_adset     = _first_existing(df, ["Ad set name", "Ad set name.1", "Ad Set", "Ad Set Name"])
    col_campaign  = _first_existing(df, ["Campaign name", "Campaign", "Campaign Name"])
    col_day       = _first_existing(df, ["Day", "date", "Date", "Reporting starts"])
    col_spend     = _first_existing(df, ["Amount spent (USD)", "Spend", "Amount Spent"])
    col_roas_w    = _first_existing(df, ["Website purchase ROAS (return on advertising spend)"])
    col_roas      = _first_existing(df, ["Purchase ROAS (return on ad spend)", "ROAS"])
    col_revenue   = _first_existing(df, ["Purchase conversion value", "Website purchase conversion value", "Revenue", "Sales"])
    col_clicks    = _first_existing(df, ["Link clicks", "Unique link clicks", "Outbound clicks", "Unique outbound clicks"])
    col_ctr       = _first_existing(df, ["CTR (link click-through rate)", "Unique link click-through rate", "Unique outbound CTR"])
    col_hook      = _first_existing(df, ["hook rate", "Hook rate", "Video hook", "3-second video plays / impressions"])
    col_hold      = _first_existing(df, ["hold rate", "Hold rate", "Video hold", "ThruPlays / impressions"])
    col_atc       = _first_existing(df, ["Adds to cart", "Website adds to cart", "Add to cart"])
    col_ci        = _first_existing(df, ["Checkouts initiated", "Website checkouts initiated", "Initiate checkout"])
    col_purch     = _first_existing(df, ["Website purchases", "Purchases", "purchases"])
    col_cpr       = _first_existing(df, ["Cost per purchase", "Cost per result"])
    col_imps      = _first_existing(df, ["Impressions"])
    col_freq      = _first_existing(df, ["Frequency"])

    out = pd.DataFrame()
    out["ad"]       = df[col_ad] if col_ad else "Unknown"
    out["adset"]    = df[col_adset] if col_adset else ""
    out["campaign"] = df[col_campaign] if col_campaign else ""
    # dates
    if col_day:
        try:
            out["dte"] = pd.to_datetime(df[col_day]).dt.date.astype(str)
        except Exception:
            out["dte"] = df[col_day].astype(str)
    else:
        out["dte"] = dt.date.today().isoformat()

    # numbers
    out["spend"]    = df[col_spend].map(_to_float) if col_spend else np.nan
    # ROAS priority: website roas > roas > revenue/spend
    roas_series = None
    if col_roas_w:
        roas_series = df[col_roas_w].map(_to_float)
    elif col_roas:
        roas_series = df[col_roas].map(_to_float)
    else:
        roas_series = pd.Series(np.nan, index=df.index)

    revenue = df[col_revenue].map(_to_float) if col_revenue else np.nan
    if isinstance(revenue, pd.Series):
        out["revenue"] = revenue
    else:
        out["revenue"] = np.nan

    out["clicks"]   = df[col_clicks].map(_to_float) if col_clicks else np.nan
    out["ctr"]      = df[col_ctr].map(_pct_to_decimal) if col_ctr else np.nan
    out["hook"]     = df[col_hook].map(_to_float) if col_hook else np.nan
    out["hold"]     = df[col_hold].map(_to_float) if col_hold else np.nan
    out["atc"]      = df[col_atc].map(_to_float) if col_atc else np.nan
    out["ic"]       = df[col_ci].map(_to_float) if col_ci else np.nan
    out["purchases"]= df[col_purch].map(_to_float) if col_purch else np.nan
    out["cpr"]      = df[col_cpr].map(_to_float) if col_cpr else np.nan
    out["imps"]     = df[col_imps].map(_to_float) if col_imps else np.nan
    out["freq"]     = df[col_freq].map(_to_float) if col_freq else np.nan

    # compute roas if missing & revenue present
    out["roas"] = roas_series
    need_roas = out["roas"].isna() & out["spend"].gt(0) & out["revenue"].notna()
    out.loc[need_roas, "roas"] = out.loc[need_roas, "revenue"] / out.loc[need_roas, "spend"]

    # compute revenue from spend*roas if revenue missing
    need_rev = out["revenue"].isna() & out["spend"].notna() & out["roas"].notna()
    out.loc[need_rev, "revenue"] = out.loc[need_rev, "spend"] * out.loc[need_rev, "roas"]

    # CVR as purchases/clicks
    with np.errstate(divide="ignore", invalid="ignore"):
        out["cvr"] = out["purchases"] / out["clicks"]

    # clean weird percentages in hook/hold (if someone exported as 35 -> 35%)
    for col in ["hook", "hold"]:
        out[col] = out[col].apply(lambda v: np.nan if pd.isna(v) else (v/100.0 if v > 1.0 else v))

    # clamp insane values to avoid skewing UI
    out["roas"] = out["roas"].clip(lower=0, upper=50)
    out["ctr"]  = out["ctr"].clip(lower=0, upper=0.5)
    out["hook"] = out["hook"].clip(lower=0, upper=1)
    out["hold"] = out["hold"].clip(lower=0, upper=1)

    return out

def _fmt_pct(x):
    try:
        txt = raw.decode("utf-8-sig", errors="ignore")
        return f"{float(x)*100:.1f}%"
    except Exception:
        txt = raw.decode("latin-1", errors="ignore")
    buf = io.StringIO(txt)
    df = pd.read_csv(buf)
    return df
        return "-"

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
GOOD = {"hook": SETTINGS["hook_good"], "hold": SETTINGS["hold_good"], "ctr": SETTINGS["ctr_good"]}

def _growth_score(row):
    roas   = float(row.get("roas", 0) or 0)
    cpr    = float(row.get("cpr", 0) or 0)
    ctr    = float(row.get("ctr", 0) or 0)
    hook   = float(row.get("hook", 0) or 0)
    hold   = float(row.get("hold", 0) or 0)
    spend  = float(row.get("spend", 0) or 0)

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
    early  = (min(ctr / GOOD["ctr"] if GOOD["ctr"]>0 else 0, 2.0) +
              min(hook / GOOD["hook"] if GOOD["hook"]>0 else 0, 2.0) +
              min(hold / GOOD["hold"] if GOOD["hold"]>0 else 0, 2.0)) / 3
    roi    = min(roas / SETTINGS["target_roas"] if SETTINGS["target_roas"]>0 else roas, 2.0)
    spendw = min(spend / SETTINGS["min_scale_spend"], 1.0) if SETTINGS["min_scale_spend"]>0 else 0
    pain   = 0 if cpr == 0 else min(1.5 / cpr, 1.0)

    return round(0.45 * roi + 0.35 * early + 0.10 * spendw + 0.10 * pain, 3)

def _ad_card(row):
    tips = []
    if float(row.get("hook",0) or 0) < GOOD["hook"]: tips.append("Strengthen 0–3s hook (pattern break).")
    if float(row.get("hold",0) or 0) < GOOD["hold"]: tips.append("Add proof or story beat @7–12s.")
    if float(row.get("ctr",0)  or 0) < GOOD["ctr"]:  tips.append("Tighter first line + clearer CTA.")
    if float(row.get("roas",0) or 0) < SETTINGS["breakeven_roas"]:
        tips.append("Below breakeven—iterate or pause.")
    if not tips: tips = ["On track—watch for 72h stability before scaling."]

    metrics = [
        {"k":"Spend", "v": f"${float(row.get('spend',0) or 0):.2f}"},
        {"k":"ROAS",  "v": f"{float(row.get('roas',0) or 0):.2f}"},
        {"k":"CPR",   "v": f"${float(row.get('cpr',0) or 0):.2f}"},
        {"k":"CVR",   "v": _fmt_pct(row.get('cvr',0))},
        {"k":"CTR",   "v": _fmt_pct(row.get('ctr',0))},
        {"k":"Hook",  "v": _fmt_pct(row.get('hook',0))},
        {"k":"Hold",  "v": _fmt_pct(row.get('hold',0))},
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return {
        "title": row.get("name") or row.get("ad") or "Ad",
        "score": _growth_score(row),
        "metrics": metrics,
        "tips": tips,
        "raw": row,
    }

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
def _coach_narrative(summary, funnel, top_cards):
    s = summary
    f = funnel.get("blended", {})
    lines = []
    lines.append(
        f"Spend ${s['spend']:.2f} on {int(s.get('purchases',0) or 0)} purchases "
        f"→ avg ROAS **{s.get('avg_roas',0):.2f}** (target {s['settings']['target_roas']})."
    )
    if f:
        lines.append(
            f"Funnel — Click→ATC: **{_fmt_pct(f.get('click_to_atc',0))}**, "
            f"ATC→IC: **{_fmt_pct(f.get('atc_to_ic',0))}**, "
            f"IC→Purchase: **{_fmt_pct(f.get('ic_to_purchase',0))}**."
        )
    if funnel.get("diagnosis"):
        lines.append("Diagnosis: " + "; ".join(funnel["diagnosis"]))
    if top_cards:
        winners = [c["title"] for c in top_cards[:3]]
        lines.append("Watching potential winners: " + ", ".join(winners))
    lines.append(
        "Next 24–72h: ensure ≥60% 7-day click before scaling +20%; "
        "shore up the biggest funnel leak with a page change + one creative variant."
    )
    return "\n".join(lines)

# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------
@app.get("/healthz")
def health():
    return {"ok": True, "has_data": LAST_DF is not None}

@app.get("/debug/last_error")
def last_error():
    return {"last_error": LAST_ERROR or ""}

@app.post("/ingest_csv_debug")
async def ingest_csv_debug(file: UploadFile = File(...)):
    """Parse-only: show columns/dtypes/sample. No write."""
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        dtypes = {c: str(df[c].dtype) for c in df.columns}
        sample = df.head(5).fillna("").astype(str).to_dict(orient="records")
        return {"columns": list(df.columns), "dtypes": dtypes, "sample": sample}
    except Exception as e:
        set_last_error(e)
        return JSONResponse({"detail": "Failed to parse CSV."}, status_code=500)

@app.post("/ingest_csv")
async def ingest_csv(file: UploadFile = File(...)):
    """Upload any Meta CSV (any date range). Rows are normalized and kept in memory."""
    global LAST_DF
    try:
        content = await file.read()
        raw = pd.read_csv(io.BytesIO(content))
        norm = _normalize_columns(raw)

    # derive effective cpr (aka CPA) if not provided
    if "cpr" not in df.columns:
        if "spend" in df.columns and "purchases" in df.columns:
            df["cpr"] = pd.to_numeric(df["spend"], errors="coerce") / pd.to_numeric(df["purchases"], errors="coerce")
        # append if we already have data
        if LAST_DF is not None:
            LAST_DF = pd.concat([LAST_DF, norm], ignore_index=True)
        else:
            df["cpr"] = pd.NA
            LAST_DF = norm

    return df
        return {
            "rows_added": int(norm.shape[0]),
            "columns": list(norm.columns),
            "note": "Stored in memory. Restart clears it.",
        }
    except Exception as e:
        set_last_error(e)
        return JSONResponse({"detail": "Server failed to ingest CSV. Hit /debug/last_error for details."}, status_code=500)

def _analyze(df: pd.DataFrame) -> dict:
    """Core analysis used by /coach."""
    if df is None or df.empty:
        return {
            "summary": {"spend": 0, "purchases": 0, "clicks": 0, "avg_roas": 0, "settings": SETTINGS},
            "funnel": {"blended": {}, "diagnosis": ["No data ingested."]},
            "decisions": {"scale": [], "iterate": [], "kill": []},
        }

    g = df.groupby("ad", dropna=False)

def _aggregate_by_ad(df: pd.DataFrame) -> pd.DataFrame:
    # Sum spend/clicks/purchases; average soft metrics by spend-weight (simple: mean)
    agg = {
    # aggregate per ad
    agg = g.agg({
        "spend": "sum",
        "revenue": "sum",
        "purchases": "sum",
        "link_clicks": "sum",
        "atc": "sum",
        "ic": "sum",
        "cpr": "mean",
        "roas": "mean",
        "clicks": "sum",
        "ctr": "mean",
        "hook_rate": "mean",
        "hold_rate": "mean",
        "cpm": "mean",
        "reach": "sum",
        "frequency": "mean",
        "hook": "mean",
        "hold": "mean",
        "cpr": "mean",
        "cvr": "mean",
    }).reset_index()

    # fallback roas if needed
    agg["roas"] = np.where(agg["spend"]>0, (agg["revenue"].fillna(0))/agg["spend"], np.nan)
    agg["name"] = agg["ad"]

    # account summary
    spend = float(df["spend"].sum(skipna=True))
    purchases = float(df["purchases"].sum(skipna=True))
    clicks = float(df["clicks"].sum(skipna=True))
    total_rev = float(df["revenue"].sum(skipna=True))
    avg_roas = (total_rev / spend) if spend > 0 else 0.0

    summary = {
        "spend": round(spend, 2),
        "purchases": int(purchases),
        "clicks": int(clicks),
        "avg_roas": round(avg_roas, 2),
        "settings": SETTINGS,
    }

    # simple funnel
    atc = float(df["atc"].sum(skipna=True)) if "atc" in df else np.nan
    ic  = float(df["ic"].sum(skipna=True)) if "ic" in df else np.nan
    def _rate(n, d): 
        if d and d>0: 
            return n / d
        return 0.0
    funnel_blended = {
        "clicks": clicks,
        "add_to_cart": atc,
        "initiate_checkout": ic,
        "purchases": purchases,
        "click_to_atc": _rate(atc, clicks),
        "atc_to_ic": _rate(ic, atc),
        "ic_to_purchase": _rate(purchases, ic),
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
    if funnel_blended["ic_to_purchase"] < 0.4 and purchases >= 5:
        diagnosis.append("Low Checkout→Purchase (checkout friction, payment issues, missing reassurance).")
    if funnel_blended["click_to_atc"] < 0.2 and clicks >= 200:
        diagnosis.append("Low Click→ATC (offer clarity, price anchoring, trust signals).")
    if not diagnosis:
        diagnosis.append("Funnel is relatively healthy; improvements likely from creative/targeting or AOV uplift.")
        diagnosis.append("No major funnel leak detected; focus on creative testing + offer depth.")

    # decisions
    SCALE, ITERATE, KILL = [], [], []
    for _, r in agg.iterrows():
        row = {
            "ad_id": r["ad"],
            "name": r["name"],
            "spend": round(float(r["spend"] or 0), 2),
            "roas": round(float(r["roas"] or 0), 3),
            "cpr": round(float(r["cpr"] or 0), 3) if not pd.isna(r["cpr"]) else 0,
            "cvr": round(float(r["cvr"] or 0), 4) if not pd.isna(r["cvr"]) else 0,
            "ctr": round(float(r["ctr"] or 0), 4) if not pd.isna(r["ctr"]) else 0,
            "hook": round(float(r["hook"] or 0), 4) if not pd.isna(r["hook"]) else 0,
            "hold": round(float(r["hold"] or 0), 4) if not pd.isna(r["hold"]) else 0,
        }
        reason = ""
        if row["spend"] >= SETTINGS["min_scale_spend"] and row["roas"] >= SETTINGS["target_roas"]:
            reason = "≥ target ROAS and enough spend (verify ≥60% from 7d-click before scaling +20%)."
            SCALE.append({**row, "reason": reason})
        elif row["spend"] >= SETTINGS["min_test_spend"] and (row["roas"] < SETTINGS["breakeven_roas"] or row["cpr"]==0 or row["roas"]==0):
            reason = "Below breakeven or no purchases at test spend."
            KILL.append({**row, "reason": reason})
        else:
            if row["spend"] < SETTINGS["min_test_spend"]:
                reason = "Not enough spend (< test threshold)."
            else:
                reason = "Between breakeven and target."
            ITERATE.append({**row, "reason": reason})

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
    decisions = {"scale": SCALE, "iterate": ITERATE, "kill": KILL}

    return {"summary": summary, "funnel": {"blended": funnel_blended, "diagnosis": diagnosis}, "decisions": decisions}

def _serp_search(q: str, num: int = 5) -> List[Dict[str, str]]:
    if not _settings.external_data:
        return []
    api_key = os.getenv("SERPAPI_KEY", "").strip()
    if not api_key:
        return [{"source": "SERPAPI", "title": "External data is OFF or missing SERPAPI_KEY", "link": "", "snippet": ""}]
@app.post("/coach")
def coach():
    try:
        params = {
            "engine": "google",
            "q": q,
            "num": num,
            "api_key": api_key,
        global LAST_DF
        analysis = _analyze(LAST_DF if LAST_DF is not None else pd.DataFrame())
        # Build UI payload
        rows_for_cards = analysis["decisions"]["iterate"] + analysis["decisions"]["scale"] + analysis["decisions"]["kill"]
        cards = [_ad_card(r) for r in rows_for_cards]
        cards.sort(key=lambda c: c["score"], reverse=True)

        ui = {
            "sections": [
                {"title": "Account Summary", "kind": "summary", "data": analysis["summary"]},
                {"title": "Funnel", "kind": "funnel", "data": analysis["funnel"]},
                {"title": "Top Candidates", "kind": "cards", "data": cards[:8]},
                {"title": "Decisions", "kind": "decisions", "data": analysis["decisions"]},
                {"title": "Risks / Diagnosis", "kind": "list", "data": analysis["funnel"]["diagnosis"]},
            ],
            "narrative": _coach_narrative(analysis["summary"], analysis["funnel"], cards[:3]),
        }

        return {
            "summary": analysis["summary"],
            "funnel": analysis["funnel"],
            "decisions": analysis["decisions"],
            "ui": ui,
            "answer": ui["narrative"],
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
        set_last_error(e)
        return JSONResponse({"detail": "Coach failed. Hit /debug/last_error for details."}, status_code=500)

@app.post("/prompt_lab")
def prompt_lab(payload: dict):
    goal = (payload.get("goal") or "").strip()
    ctx  = (payload.get("context") or "").strip()
    if not goal:
        return {"error": "Missing goal"}
    base = f"""Act as a senior performance creative strategist.
Goal: {goal}
Audience/Context: {ctx or 'general DTC'}.
Constraints: Meta policy-safe, short lines, benefits-first, screen-native.
Deliverables:
1) 10 hooks (≤8 words) labeled Pain/Curiosity/Proof/Social.
2) 3 concept outlines (beats + on-screen text).
3) 3 CTA lines that reduce friction.
4) 5 landing-page trust blocks if drop-off is post-click.
Return in clean markdown sections."""
    alternates = [
        "Make a 25s Reels VSL using problem→solution→proof→CTA. Include 3 pattern interrupts.",
        "Write 5 primary texts (≤125 chars) + 5 headlines (≤40 chars) optimizing for lower CPR.",
        "Draft a landing page hero (H1, sub, bullets, proof bar, CTA) for colder traffic."
    ]
    return {"goal": goal, "context": ctx, "prompt": base, "alternates": alternates}

@app.post("/script_doctor")
def script_doctor(payload: dict):
    txt = (payload.get("script") or "").strip()
    if not txt:
        return {"error": "Missing script text"}

    # Simple heuristics — upgrade later with NLP
    lower = txt.lower()
    segments = [
        {"label":"Hook",    "ok": lower.startswith(("how","why","what","stop","warning","women","men")) or "?" in lower.split("\n",1)[0]},
        {"label":"Problem", "ok": any(k in lower for k in ["pain","struggle","problem","issue"])},
        {"label":"Agitate", "ok": any(k in lower for k in ["worse","keeps","tired of","fed up"])},
        {"label":"Solution","ok": any(k in lower for k in ["it's because","solution","here's how","works"])},
        {"label":"Proof",   "ok": any(k in lower for k in ["proof","testimonial","before","after","clinically","study"])},
        {"label":"Offer",   "ok": any(k in lower for k in ["today","save","bundle","guarantee","free"])},
        {"label":"CTA",     "ok": any(k in lower for k in ["shop now","get started","try it","learn more","see options","order now"])},
    ]
    tips = [
        "Add a crisp 1-line hook in the first 2s with a pattern break.",
        "Insert quick proof (testimonial, number, before/after) by 8–12s.",
        "End with a direct CTA + micro-urgency (‘See options’).",
        "If CPR high but CTR fine, tighten product promise; if CTR low, rewrite first line & visual opener."
    ]
    shotlist = [
        {"t":"0–3s","shot":"Close-up","text":"Pain question","on_screen":"Large subtitle"},
        {"t":"4–8s","shot":"Waist + product","text":"Reveal solution","on_screen":"Benefit 1"},
        {"t":"9–15s","shot":"B-roll + overlay","text":"Proof/demo","on_screen":"‘Real users’"},
        {"t":"16–22s","shot":"Medium","text":"Offer + CTA","on_screen":"Try risk-free"},
    ]
    return {"segments": segments, "tips": tips, "shotlist": shotlist}

# --------------------------------------------------------------------------------------
# Minimal dark UI (Tailwind) — renders Coach cards, Prompt Lab, Script Doctor
# --------------------------------------------------------------------------------------
INDEX_HTML = """
<!doctype html>
<html>
<html lang="en">
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
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ERTO Agent</title>
<script src="https://cdn.tailwindcss.com"></script>
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
<body class="bg-[#0b1220] text-slate-100">
<div class="max-w-6xl mx-auto p-4 space-y-6">
  <header class="flex items-center justify-between">
    <h1 class="text-2xl font-semibold">ERTO Agent</h1>
    <nav class="flex gap-2 text-sm">
      <button data-tab="ingest"  class="tab px-3 py-1 rounded bg-slate-800">Ingest</button>
      <button data-tab="coach"   class="tab px-3 py-1 rounded bg-slate-700">Coach</button>
      <button data-tab="prompt"  class="tab px-3 py-1 rounded bg-slate-800">Prompt Lab</button>
      <button data-tab="script"  class="tab px-3 py-1 rounded bg-slate-800">Script Doctor</button>
    </nav>
  </header>

  <section id="coach" class="hidden space-y-4">
    <div id="coach-narrative" class="bg-slate-900/60 rounded-xl p-4 text-slate-200"></div>
    <div id="coach-sections" class="space-y-4"></div>
  </section>

  <section id="prompt" class="hidden">
    <form id="prompt-form" class="bg-slate-900/60 rounded-xl p-4 space-y-3">
      <input name="goal" class="w-full rounded bg-slate-800 p-2" placeholder="Goal (e.g., 25s VSL for RLT belt)"/>
      <input name="context" class="w-full rounded bg-slate-800 p-2" placeholder="Context (audience, tone, constraints)"/>
      <button class="rounded bg-indigo-600 hover:bg-indigo-500 px-3 py-2">Generate</button>
    </form>
    <div id="prompt-out" class="mt-4 grid gap-4"></div>
  </section>

  <section id="script" class="hidden">
    <form id="script-form" class="bg-slate-900/60 rounded-xl p-4 space-y-3">
      <textarea name="script" rows="6" class="w-full rounded bg-slate-800 p-2" placeholder="Paste script…"></textarea>
      <button class="rounded bg-emerald-600 hover:bg-emerald-500 px-3 py-2">Analyze</button>
    </form>
    <div id="script-out" class="mt-4 grid gap-4"></div>
  </section>

  <section id="ingest" class="">
    <div class="rounded-xl bg-slate-900/60 p-4">
      <div class="text-slate-300">Upload your CSV in <a class="underline" href="/docs" target="_blank">/docs</a> → <b>POST /ingest_csv</b>, then click the button below.</div>
      <button id="refresh-coach" class="mt-3 rounded bg-indigo-600 px-3 py-2">Refresh Coach</button>
      <div id="health" class="mt-2 text-slate-400 text-sm"></div>
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
  </section>
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
tabs.forEach(b => b.onclick = () => {
  document.querySelectorAll('section').forEach(s => s.classList.add('hidden'));
  document.querySelectorAll('.tab').forEach(t => t.classList.replace('bg-slate-700','bg-slate-800'));
  document.getElementById(b.dataset.tab).classList.remove('hidden');
  b.classList.replace('bg-slate-800','bg-slate-700');
});

async function fetchCoach(){
  const res = await fetch('/coach', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({})});
  const data = await res.json();
  if(data.ui){ renderCoach(data.ui); }
  else { document.getElementById('coach-narrative').innerText = 'No data yet. Ingest a CSV first.'; }
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
function chip(k,v){ return `<div class="px-2 py-1 rounded bg-slate-800 text-sm"><span class="text-slate-400">${k}</span> <span class="ml-1 font-medium">${v}</span></div>`;}
function card(c){
  const ms = (c.metrics||[]).map(m => chip(m.k, m.v)).join('');
  const tips = (c.tips||[]).map(t => `<li>${t}</li>`).join('');
  return `
  <div class="rounded-2xl bg-slate-900/70 p-4 border border-slate-800">
    <div class="flex items-center justify-between">
      <div class="text-lg font-semibold">${c.title}</div>
      <div class="text-sm px-2 py-1 rounded bg-indigo-600/20 border border-indigo-600/30">Score ${c.score}</div>
    </div>
    <div class="mt-3 flex flex-wrap gap-2">${ms}</div>
    <div class="mt-3 text-slate-300">
      <div class="text-slate-400 text-sm mb-1">Tips</div>
      <ul class="list-disc ml-5">${tips}</ul>
    </div>
  </div>`;
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
function sectionBlock(s){
  if(s.kind === 'summary'){
    const a = s.data;
    return `<div class="rounded-xl bg-slate-900/60 p-4 grid sm:grid-cols-2 gap-2">
      ${chip("Spend", `$${(a.spend||0).toFixed(2)}`)}
      ${chip("Purchases", a.purchases||0)}
      ${chip("Avg ROAS", (a.avg_roas||0).toFixed(2))}
      ${chip("Target ROAS", a.settings?.target_roas||'-')}
    </div>`;
  }
  if(s.kind === 'funnel'){
    const f = s.data.blended || {};
    return `<div class="rounded-xl bg-slate-900/60 p-4">
      <div class="text-slate-300 mb-2">Funnel</div>
      <div class="flex flex-wrap gap-2">
        ${chip("Click→ATC", ((f.click_to_atc||0)*100).toFixed(1)+"%")}
        ${chip("ATC→IC", ((f.atc_to_ic||0)*100).toFixed(1)+"%")}
        ${chip("IC→Purchase", ((f.ic_to_purchase||0)*100).toFixed(1)+"%")}
      </div>
    </div>`;
  }
  if(s.kind === 'cards'){
    return `<div class="grid md:grid-cols-2 gap-3">${(s.data||[]).map(card).join('')}</div>`;
  }
  if(s.kind === 'decisions'){
    const d = s.data||{};
    function list(name, arr){
      return `<div class="rounded-xl bg-slate-900/60 p-4">
        <div class="font-medium mb-2">${name}</div>
        <ul class="list-disc ml-5 text-slate-300">${(arr||[]).slice(0,8).map(x=>`<li>${x.name||x.ad_id} — ${x.reason||''}</li>`).join('')}</ul>
      </div>`;
    }
    return `<div class="grid md:grid-cols-3 gap-3">
      ${list("Scale", d.scale)}
      ${list("Iterate", d.iterate)}
      ${list("Kill", d.kill)}
    </div>`;
  }
  if(s.kind === 'list'){
    return `<div class="rounded-xl bg-amber-900/20 border border-amber-700/30 p-4">
      <div class="font-medium mb-2">Risks / Diagnosis</div>
      <ul class="list-disc ml-5">${(s.data||[]).map(x=>`<li>${x}</li>`).join('')}</ul>
    </div>`;
  }
  return "";
}
async function toggleExternal(){
  const r = await fetch('/settings/toggle_external', {method:'POST'});
  document.getElementById('settings_out').textContent = await r.text();
  await getSettings();
function renderCoach(ui){
  document.getElementById('coach').classList.remove('hidden');
  document.querySelector('[data-tab="coach"]').classList.replace('bg-slate-800','bg-slate-700');
  document.getElementById('coach-narrative').innerHTML = `<div class="prose prose-invert">${(ui.narrative||'').replace(/\\n/g,'<br/>')}</div>`;
  document.getElementById('coach-sections').innerHTML = (ui.sections||[]).map(sectionBlock).join('');
}
getSettings();
document.getElementById('refresh-coach').onclick = async ()=>{
  const h = await fetch('/healthz').then(r=>r.json());
  document.getElementById('health').innerText = h.has_data ? 'Data loaded — open Coach.' : 'No data yet.';
  fetchCoach();
};

// Prompt Lab
document.getElementById('prompt-form').onsubmit = async (e)=>{
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = Object.fromEntries(fd.entries());
  const res = await fetch('/prompt_lab',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data = await res.json();
  const out = document.getElementById('prompt-out');
  out.innerHTML = `
    <div class="rounded-xl bg-slate-900/60 p-4">
      <div class="text-slate-400 text-sm mb-2">Goal</div>
      <div>${data.goal||''}</div>
    </div>
    <div class="rounded-xl bg-slate-900/60 p-4">
      <div class="text-slate-400 text-sm mb-2">Main Prompt</div>
      <pre class="whitespace-pre-wrap text-slate-200">${data.prompt||''}</pre>
    </div>
    <div class="rounded-xl bg-slate-900/60 p-4">
      <div class="text-slate-400 text-sm mb-2">Alternates</div>
      <ul class="list-disc ml-5">${(data.alternates||[]).map(x=>`<li>${x}</li>`).join('')}</ul>
    </div>`;
};

// Script Doctor
document.getElementById('script-form').onsubmit = async (e)=>{
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = Object.fromEntries(fd.entries());
  const res = await fetch('/script_doctor',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d = await res.json();
  const out = document.getElementById('script-out');
  out.innerHTML = `
    <div class="rounded-xl bg-slate-900/60 p-4">
      <div class="font-medium mb-2">Structure</div>
      <div class="flex flex-wrap gap-2">
        ${(d.segments||[]).map(s=>`<div class="px-2 py-1 rounded ${s.ok?'bg-emerald-900/30 border-emerald-700/30':'bg-slate-800 border-slate-700'} border">${s.label}</div>`).join('')}
      </div>
    </div>
    <div class="rounded-xl bg-slate-900/60 p-4">
      <div class="font-medium mb-2">Tips</div>
      <ul class="list-disc ml-5">${(d.tips||[]).map(x=>`<li>${x}</li>`).join('')}</ul>
    </div>
    <div class="rounded-xl bg-slate-900/60 p-4">
      <div class="font-medium mb-2">Shot-List</div>
      <div class="grid md:grid-cols-2 gap-3">
        ${(d.shotlist||[]).map(s=>`<div class="rounded bg-slate-800 p-3">
            <div class="text-slate-400 text-sm">${s.t}</div>
            <div class="font-medium">${s.shot}</div>
            <div class="text-slate-300">${s.text}</div>
            <div class="text-slate-400 text-sm">On-screen: ${s.on_screen}</div>
        </div>`).join('')}
      </div>
    </div>`;
};

// auto-load health & coach on first view
fetch('/healthz').then(r=>r.json()).then(h=>{
  document.getElementById('health').innerText = h.has_data ? 'Data loaded — open Coach.' : 'No data yet.';
});
fetchCoach();
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
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)

# ----------------------------
# Healthcheck (Render)
# ----------------------------
@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")
