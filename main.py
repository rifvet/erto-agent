# main.py
# FastAPI Media-Buyer Assistant — Coach / Prompt Lab / Script Doctor
# Drop-in single file. Requires: fastapi, uvicorn, pydantic, pandas, numpy, httpx, python-multipart

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io, os, traceback, json
import httpx
from statistics import pstdev
from datetime import datetime, timedelta

app = FastAPI(title="Erto Agent", version="2.1.0")

# ---------------------------
# In-memory store
# ---------------------------
MEMORY = {
    "df": None,
    "last_error": None,
}

# ---------------------------
# Settings (editable)
# ---------------------------
SETTINGS = {
    # Financial guardrails
    "breakeven_roas": 1.54,        # used if we can't derive breakeven CPA
    "target_roas": 2.0,
    "min_test_spend": 20.0,        # spend threshold to judge a creative
    "min_scale_spend": 50.0,       # spend threshold to consider scaling
    "avg_margin": 0.30,            # used to derive breakeven CPA from AOV when target_cpa is None
    "target_cpa": None,            # optional override (Meta CPR ~ CPA)

    # Creative soft metrics
    "hook_good": 0.30,             # 30%
    "hold_good": 0.10,             # 10%
    "ctr_good": 0.015,             # 1.5%

    # Trend & fatigue
    "trend_window": 3,             # days
    "fatigue_freq": 3.0,           # avg frequency threshold
    "fatigue_ctr_drop": 0.25,      # ≥25% CTR drop vs prior period

    # External data
    "external_data": False,        # set True to allow Serper snippets
}

# ============================================================
# Helpers
# ============================================================

def _set_error(e: Exception):
    MEMORY["last_error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))

def _read_csv(upload: UploadFile) -> pd.DataFrame:
    raw = upload.file.read()
    data = io.BytesIO(raw)
    try:
        df = pd.read_csv(data)
    except Exception:
        data.seek(0)
        df = pd.read_csv(data, encoding="utf-8", engine="python")
    return df

def _num(x):
    """Robust numeric coercion for messy CSVs."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    x = str(x).replace(",", "").strip()
    if x == "" or x.lower() in {"nan", "none", "null", "-"}:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def _as_frac_from_percent(val):
    """Convert a percent-like value (e.g., 2.5 means 2.5%) into fraction."""
    v = _num(val)
    if pd.isna(v):
        return np.nan
    # If value >1 it's likely a percentage (e.g., 3.5 => 0.035)
    return v/100.0 if v > 1 else v

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Meta export columns → normalized schema used by analyzer."""
    d = df.copy()

    # Flexible column lookup
    def pick(*names, default=None):
        for n in names:
            if n in d.columns:
                return n
        return default

    # Column candidates
    col_ad        = pick("Ad name", "Ad", "Ad Name", "ad", default=None)
    col_date      = pick("Day", "Date", "date", "Reporting starts", default=None)
    col_spend     = pick("Amount spent (USD)", "Spend", "spend", default=None)
    col_roas      = pick("Purchase ROAS (return on ad spend)", "Website purchase ROAS (return on advertising spend)", "ROAS", "roas", default=None)
    col_rev       = pick("Purchase conversion value", "Website purchase conversion value", "revenue", "Revenue", default=None)
    col_purch     = pick("Website purchases", "Purchases", "purchases", default=None)
    col_clicks    = pick("Link clicks", "Unique link clicks", "Clicks", "clicks", default=None)
    col_ctr       = pick("CTR (link click-through rate)", "Unique link click-through rate", "CTR", "ctr", default=None)
    col_hook      = pick("hook rate", "Video 3-second plays / Impressions", "hook", default=None)
    col_hold      = pick("hold rate", "Video throughplays / Impressions", "hold", default=None)
    col_cpr       = pick("Cost per purchase", "Cost per result", "CPR", "cpr", default=None)
    col_cvr       = pick("conversion rate", "Conversion Rate", "CVR", "cvr", default=None)
    col_atc       = pick("Website adds to cart", "Adds to cart", "add to cart", "atc", default=None)
    col_ic        = pick("Website checkouts initiated", "Checkouts initiated", "ic", default=None)
    col_freq      = pick("Frequency", "freq", default=None)
    col_impr      = pick("Impressions", "impressions", default=None)
    col_reach     = pick("Reach", "reach", default=None)
    col_cpm       = pick("CPM (cost per 1,000 impressions)", "CPM", "cpm", default=None)

    out = pd.DataFrame()
    out["ad"] = d[col_ad] if col_ad else "unknown"
    # dates
    if col_date:
        out["dte"] = pd.to_datetime(d[col_date], errors="coerce")
    elif "Reporting starts" in d.columns:
        out["dte"] = pd.to_datetime(d["Reporting starts"], errors="coerce")
    else:
        out["dte"] = pd.NaT

    # spend/revenue/purchases
    out["spend"] = d[col_spend].map(_num) if col_spend else np.nan
    if col_rev:
        out["revenue"] = d[col_rev].map(_num)
    else:
        # derive from ROAS if present
        if col_roas and col_spend:
            out["revenue"] = d[col_roas].map(_num) * d[col_spend].map(_num)
        else:
            out["revenue"] = np.nan
    out["purchases"] = d[col_purch].map(_num) if col_purch else np.nan

    # clicks / ctr
    out["clicks"] = d[col_clicks].map(_num) if col_clicks else np.nan
    if col_ctr:
        out["ctr"] = d[col_ctr].map(_as_frac_from_percent)
    else:
        # derive if possible later via clicks/impr
        out["ctr"] = np.nan

    # hook / hold — Meta export already fractional in many accounts
    out["hook"] = d[col_hook].map(_num) if col_hook else np.nan
    out["hold"] = d[col_hold].map(_num) if col_hold else np.nan

    # CPR, CVR
    out["cpr"] = d[col_cpr].map(_num) if col_cpr else np.nan
    if col_cvr:
        # can be % or fraction depending on custom metric; normalize
        out["cvr"] = d[col_cvr].map(_as_frac_from_percent)
    else:
        # fallback: purchases / clicks
        out["cvr"] = np.where((out["purchases"] > 0) & (out["clicks"] > 0), out["purchases"] / out["clicks"], np.nan)

    # Funnel
    out["atc"] = d[col_atc].map(_num) if col_atc else np.nan
    out["ic"]  = d[col_ic].map(_num)  if col_ic  else np.nan

    # Reach / Impr / Freq / CPM
    out["freq"] = d[col_freq].map(_num) if col_freq else np.nan
    out["impr"] = d[col_impr].map(_num) if col_impr else np.nan
    out["reach"] = d[col_reach].map(_num) if col_reach else np.nan
    out["cpm"] = d[col_cpm].map(_num) if col_cpm else np.nan

    # derive missing CTR, Impressions
    out["impr"] = np.where(out["impr"].isna() & out["reach"].notna() & out["freq"].notna(), out["reach"] * out["freq"], out["impr"])
    out["ctr"]  = np.where(out["ctr"].isna() & out["clicks"].notna() & out["impr"].notna() & (out["impr"]>0), out["clicks"] / out["impr"], out["ctr"])

    # Clean types
    out["ad"] = out["ad"].astype(str)
    return out

def _derive_cpa_targets(df: pd.DataFrame):
    """Return (breakeven_cpa, target_cpa). If target_cpa is set, use it; else derive from AOV * margin."""
    if SETTINGS.get("target_cpa"):
        return SETTINGS["target_cpa"], SETTINGS["target_cpa"]
    purchases = float(df["purchases"].sum(skipna=True) or 0)
    revenue   = float(df["revenue"].sum(skipna=True) or 0)
    aov = (revenue / purchases) if purchases > 0 else 0
    breakeven = aov * SETTINGS.get("avg_margin", 0.30) if aov > 0 else None
    return breakeven, breakeven

def _last_n_vs_prev(df: pd.DataFrame, n: int):
    if "dte" not in df.columns or df.empty:
        return df, pd.DataFrame()
    d = df.dropna(subset=["dte"]).sort_values("dte")
    if d.empty:
        return d, pd.DataFrame()
    last_day = d["dte"].max()
    last_n = d[d["dte"] > last_day - pd.Timedelta(days=n)]
    prev_n = d[(d["dte"] <= last_day - pd.Timedelta(days=1)) & (d["dte"] > last_day - pd.Timedelta(days=2*n))]
    return last_n, prev_n

def _period_agg(d):
    if d.empty:
        return {"roas":0,"cpr":0,"ctr":0,"spend":0}
    spend = float(d["spend"].sum(skipna=True) or 0)
    rev   = float(d["revenue"].sum(skipna=True) or 0)
    roas  = rev/spend if spend>0 else 0
    cpr   = (spend / float(d["purchases"].sum(skipna=True))) if float(d["purchases"].sum(skipna=True))>0 else 0
    ctr   = float(d["ctr"].mean(skipna=True) or 0)
    return {"roas":roas,"cpr":cpr,"ctr":ctr,"spend":spend}

def _roas_volatility(d):
    vals = d.dropna(subset=["roas"]).sort_values("dte")["roas"].tail(7).tolist() if "roas" in d else []
    return pstdev(vals) if len(vals) >= 2 else 0.0

async def _serper_snippets(query: str):
    if not SETTINGS.get("external_data"):
        return []
    key = os.getenv("SERPER_API_KEY")
    if not key:
        return []
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.post("https://google.serper.dev/search",
                headers={"X-API-KEY": key, "Content-Type":"application/json"},
                json={"q": query, "num": 5})
            j = r.json()
        snippets = []
        for item in (j.get("organic",[]) or [])[:5]:
            s = item.get("snippet") or ""
            if s: snippets.append(s.strip())
        return snippets
    except Exception:
        return []

def _coach_narrative(summary, funnel, top_cards):
    parts = []
    parts.append(f"Spend ${summary['spend']:.2f} on {summary['purchases']} purchases → avg ROAS **{summary['avg_roas']:.2f}** (target {SETTINGS['target_roas']}).")
    f = funnel.get("blended", {})
    if f:
        parts.append(f"Funnel — Click→ATC: **{f.get('click_to_atc',0):.1%}**, ATC→IC: **{f.get('atc_to_ic',0):.1%}**, IC→Purchase: **{f.get('ic_to_purchase',0):.1%}**.")
    dx = funnel.get("diagnosis", [])
    if dx:
        parts.append("Diagnosis: " + "; ".join(dx))
    if top_cards:
        watch = ", ".join([c["name"] for c in top_cards[:3]])
        parts.append(f"Watching potential winners: {watch}")
    parts.append("Next 24–72h: ensure ≥60% 7-day click before any +20% scale; fix the biggest funnel leak; add 1 new creative variant.")
    return " ".join(parts)

def _ad_tips(row, breakeven_cpa):
    tips = []
    # Soft metrics guidance
    if not pd.isna(row.get("hook")) and row["hook"] < SETTINGS["hook_good"]:
        tips.append("Hook is soft — test a harder pattern break in first 2s (motion + contrast + direct benefit).")
    if not pd.isna(row.get("hold")) and row["hold"] < SETTINGS["hold_good"]:
        tips.append("Hold is weak — tighten middle: demo faster, add captions & quick proof within 8–12s.")
    if not pd.isna(row.get("ctr")) and row["ctr"] < SETTINGS["ctr_good"]:
        tips.append("CTR low — thumbnail/first-frame and opening line need stronger curiosity or outcome.")
    # Down-funnel diagnosis
    if row.get("cvr") and row["cvr"] < 0.01 and (row.get("ctr") or 0) >= SETTINGS["ctr_good"]:
        tips.append("Clicks not converting — suspect PDP clarity/price anchor; add social proof bar + FAQs near CTA.")
    # CPR / ROAS gates
    if breakeven_cpa and row.get("cpa") and row["cpa"] > breakeven_cpa * 1.1:
        tips.append(f"CPA ${row['cpa']:.2f} above breakeven — pause or re-cut creative; try stronger offer framing.")
    if row.get("roas",0) >= SETTINGS["target_roas"] and row.get("spend",0) >= SETTINGS["min_scale_spend"]:
        tips.append("On track — verify ≥60% 7-day click attribution and scale +20%.")
    if not tips:
        tips.append("Between breakeven and target — let it gather more spend or iterate a variant.")
    return tips

def _analyze(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "summary": {"spend": 0, "purchases": 0, "clicks": 0, "avg_roas": 0, "settings": SETTINGS},
            "funnel": {"blended": {}, "diagnosis": ["No data ingested."]},
            "decisions": {"scale": [], "iterate": [], "kill": []},
            "rising": [], "fatigue": [], "playbook": []
        }

    # Account-level summary
    spend = float(df["spend"].sum(skipna=True))
    purchases = float(df["purchases"].sum(skipna=True))
    clicks = float(df["clicks"].sum(skipna=True))
    revenue = float(df["revenue"].sum(skipna=True))
    avg_roas = (revenue/spend) if spend>0 else 0.0
    breakeven_cpa, target_cpa = _derive_cpa_targets(df)

    summary = {
        "spend": round(spend,2),
        "purchases": int(purchases),
        "clicks": int(clicks),
        "avg_roas": round(avg_roas,2),
        "settings": {**SETTINGS, "breakeven_cpa": breakeven_cpa, "target_cpa": target_cpa}
    }

    # Per-ad aggregates
    g = df.groupby("ad", dropna=False)
    agg = g.agg({
        "spend":"sum","revenue":"sum","purchases":"sum","clicks":"sum",
        "ctr":"mean","hook":"mean","hold":"mean","cpr":"mean","cvr":"mean","freq":"mean"
    }).reset_index().rename(columns={"ad":"name"})
    agg["roas"] = np.where(agg["spend"]>0, (agg["revenue"].fillna(0))/agg["spend"], 0)
    agg["cpa"]  = np.where(agg["purchases"]>0, agg["spend"]/agg["purchases"], 0)

    # Funnel diagnosis
    atc = float(df["atc"].sum(skipna=True)) if "atc" in df else np.nan
    ic  = float(df["ic"].sum(skipna=True))  if "ic" in df  else np.nan
    def r(n,d): return (n/d) if (d and d>0) else 0.0
    funnel_blended = {
        "clicks": clicks, "add_to_cart": atc, "initiate_checkout": ic, "purchases": purchases,
        "click_to_atc": r(atc, clicks), "atc_to_ic": r(ic, atc), "ic_to_purchase": r(purchases, ic)
    }
    diagnosis = []
    if funnel_blended["click_to_atc"] < 0.20 and clicks>=200:
        diagnosis.append("Low Click→ATC: clarify promise & price anchor on PDP; make the ‘why now’ obvious above the fold.")
    if funnel_blended["atc_to_ic"] < 0.40 and atc>=80:
        diagnosis.append("Low ATC→IC: cart UX friction or fee surprise; make ‘Continue to checkout’ primary, surface shipping earlier.")
    if funnel_blended["ic_to_purchase"] < 0.40 and ic>=40:
        diagnosis.append("Low Checkout→Purchase: payment trust issues; add badges by button, returns copy, express options (ShopPay/PayPal).")
    if not diagnosis:
        diagnosis.append("No single catastrophic leak; focus on creative/offer depth and stable scaling rhythm (72h).")

    # Rising winners & fatigue (window vs previous window)
    rising, fatigue = [], []
    n = SETTINGS["trend_window"]
    for ad, d_ad in df.groupby("ad"):
        d_ad = d_ad.copy()
        d_ad["roas"] = np.where(d_ad["spend"]>0, d_ad["revenue"]/d_ad["spend"], np.nan)
        last_n, prev_n = _last_n_vs_prev(d_ad, n)
        cur = _period_agg(last_n); prev = _period_agg(prev_n)
        if cur["spend"]>=SETTINGS["min_test_spend"] and cur["roas"]>max(prev["roas"], SETTINGS["breakeven_roas"])*1.15:
            rising.append({"name": ad, "roas_now": round(cur["roas"],2), "roas_prev": round(prev["roas"],2), "spend_now": round(cur["spend"],2)})
        freq = float(d_ad["freq"].mean(skipna=True) or 0)
        ctr_now, ctr_prev = cur["ctr"], prev["ctr"]
        if freq >= SETTINGS["fatigue_freq"] and ctr_prev>0 and (ctr_prev-ctr_now)/ctr_prev >= SETTINGS["fatigue_ctr_drop"]:
            fatigue.append({"name": ad, "freq": round(freq,2), "ctr_now": round(ctr_now,4), "ctr_prev": round(ctr_prev,4)})

    # Build decisions with richer reasons
    SCALE, ITERATE, KILL = [], [], []
    for _, r in agg.sort_values("spend", ascending=False).iterrows():
        row = {
            "ad_id": r["name"], "name": r["name"],
            "spend": round(float(r["spend"] or 0),2),
            "roas": round(float(r["roas"] or 0),3),
            "cpa": round(float(r["cpa"] or 0),2),
            "cpr": round(float(r["cpr"] or 0),3) if not pd.isna(r["cpr"]) else 0,
            "cvr": round(float(r["cvr"] or 0),4) if not pd.isna(r["cvr"]) else 0,
            "ctr": round(float(r["ctr"] or 0),4) if not pd.isna(r["ctr"]) else 0,
            "hook": round(float(r["hook"] or 0),4) if not pd.isna(r["hook"]) else 0,
            "hold": round(float(r["hold"] or 0),4) if not pd.isna(r["hold"]) else 0,
            "freq": round(float(r["freq"] or 0),2) if not pd.isna(r["freq"]) else 0,
        }
        # volatility by ad
        d_ad = df[df["ad"]==r["name"]].copy()
        d_ad["roas"] = np.where(d_ad["spend"]>0, d_ad["revenue"]/d_ad["spend"], np.nan)
        vol = _roas_volatility(d_ad)
        tips = _ad_tips(row, breakeven_cpa)

        # scale?
        if row["spend"]>=SETTINGS["min_scale_spend"] and row["roas"]>=SETTINGS["target_roas"]:
            reason = (f"Meets scaling gate: ROAS {row['roas']:.2f} ≥ target {SETTINGS['target_roas']} with spend ${row['spend']:.0f}. "
                      f"Volatility last 7d: {vol:.2f}. Action: +20% if ≥60% 7-day click attribution; recheck in 24–48h.")
            SCALE.append({**row, "reason": reason, "tips": tips})
        # kill?
        elif row["spend"]>=SETTINGS["min_test_spend"] and (
             row["roas"] < SETTINGS["breakeven_roas"] or (breakeven_cpa and row["cpa"]>breakeven_cpa*1.1)):
            reason = ("Below breakeven after test spend — either ROAS under breakeven "
                      f"or CPA ${row['cpa']:.2f} above allowable. Action: pause or cut and re-enter with re-cut creative/offer.")
            KILL.append({**row, "reason": reason, "tips": tips})
        else:
            reason = ("Not enough evidence yet or between breakeven and target. "
                      "Action: keep gathering data or iterate a small variant (first 2s, headline, price anchor).")
            ITERATE.append({**row, "reason": reason, "tips": tips})

    # Risks (account-level)
    risks = []
    # High frequency + decaying CTR across account
    if float(df["freq"].mean(skipna=True) or 0) >= SETTINGS["fatigue_freq"]:
        risks.append("Rising frequency across account — expect creative fatigue; introduce 1–2 fresh intros this week.")
    # CPM
    if "cpm" in df.columns and float(df["cpm"].mean(skipna=True) or 0) > 30:
        risks.append("Elevated CPMs — broaden audiences or refresh creatives that earn better watch-time to win cheaper delivery.")
    # Spend concentration
    spend_share = agg.set_index("name")["spend"].sort_values(ascending=False)
    if not spend_share.empty and spend_share.iloc[0] > 0.5 * spend_share.sum():
        risks.append(f"Spend concentration — '{spend_share.index[0]}' holds >50% of spend; diversify with 1–2 near clones.")
    if not risks:
        risks.append("No critical risks flagged; stick to rhythm: daily check, 72h rule, small iterative launches.")

    # Playbook
    playbook = []
    if diagnosis:
        dx = diagnosis[0]
        if "Checkout" in dx:
            playbook.append({"do":"Checkout trust sprint","impact":"High","effort":"Low",
                             "steps":["Add card/PayPal/ShopPay badges near button",
                                      "Inline returns & shipping below CTA",
                                      "Enable express checkout",
                                      "Reduce fields and auto-fill where possible"]})
        elif "ATC→IC" in dx:
            playbook.append({"do":"Cart → Checkout clarity","impact":"Med","effort":"Low",
                             "steps":["Surface shipping/fees in cart",
                                      "Make 'Continue to checkout' the primary action",
                                      "Remove visual clutter/distractions"]})
        elif "Click→ATC" in dx:
            playbook.append({"do":"PDP above-the-fold tighten","impact":"Med","effort":"Med",
                             "steps":["Rewrite promise to outcome",
                                      "Add social proof bar (stars + count)",
                                      "Price anchor with value bullets and micro-urgency"]})
    for w in SCALE[:2] or rising[:2]:
        playbook.append({"do":f"Budget +20% on '{w['name']}'","impact":"High","effort":"Low",
                         "steps":["Verify ≥60% 7-day click",
                                  "Watch CPR/AOV for 24–48h",
                                  "If stable, repeat +20% tomorrow"]})

    return {
        "summary": summary,
        "funnel": {"blended": funnel_blended, "diagnosis": diagnosis, "risks": risks},
        "decisions": {"scale": SCALE, "iterate": ITERATE, "kill": KILL},
        "rising": rising, "fatigue": fatigue, "playbook": playbook
    }

# ============================================================
# Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML

@app.post("/ingest_csv")
def ingest_csv(file: UploadFile = File(...)):
    try:
        raw = _read_csv(file)
        df = _standardize_columns(raw)
        MEMORY["df"] = df
        return {"status": "ok", "rows": len(df), "columns": list(df.columns)}
    except Exception as e:
        _set_error(e)
        return JSONResponse(status_code=500, content={"detail":"Server failed to ingest CSV. Hit /debug/last_error for details."})

@app.post("/ingest_csv_debug")
def ingest_csv_debug(file: UploadFile = File(...)):
    try:
        raw = _read_csv(file)
        sample = raw.head(5).fillna("").astype(str).to_dict(orient="records")
        dtypes = {c:str(t) for c,t in raw.dtypes.to_dict().items()}
        return {"columns": list(raw.columns), "dtypes": dtypes, "sample": sample}
    except Exception as e:
        _set_error(e)
        return JSONResponse(status_code=500, content={"detail":"Could not parse CSV. See /debug/last_error."})

@app.get("/coach")
async def coach():
    try:
        df = MEMORY["df"]
        analysis = _analyze(df)
        # Optional: external snippets for the top leak
        leak = analysis["funnel"]["diagnosis"][0] if analysis["funnel"]["diagnosis"] else ""
        snippets = await _serper_snippets(f"ecommerce {leak or 'checkout conversion'} quick wins")
        cards = []
        for bucket in ("scale","iterate","kill"):
            for r in analysis["decisions"][bucket][:8]:
                cards.append({
                    "name": r["name"],
                    "score": round((r["roas"] / max(SETTINGS["target_roas"], 0.01)) + (r["spend"]/100.0), 3),
                    "spend": r["spend"], "roas": r["roas"], "cpr": r["cpa"] or r["cpr"],
                    "cvr": r["cvr"], "ctr": r["ctr"], "hook": r["hook"], "hold": r["hold"],
                    "bucket": bucket, "tips": r.get("tips",[]), "reason": r.get("reason","")
                })
        ui = {
            "sections": [
                {"title":"Account Summary","kind":"summary","data": analysis["summary"]},
                {"title":"Funnel","kind":"funnel","data": analysis["funnel"]},
                {"title":"Top Candidates","kind":"cards","data": cards[:12]},
                {"title":"Rising Winners","kind":"list","data":[f"{r['name']} — ROAS {r['roas_now']:.2f} (prev {r['roas_prev']:.2f})" for r in analysis.get("rising",[])]},
                {"title":"Fatigue Watch","kind":"list","data":[f"{f['name']} — freq {f['freq']}, CTR now {f['ctr_now']:.2%} (prev {f['ctr_prev']:.2%})" for f in analysis.get('fatigue',[])]},
                {"title":"Decisions","kind":"decisions","data": analysis["decisions"]},
                {"title":"Playbook (Next Moves)","kind":"playbook","data": analysis.get("playbook",[])},
                {"title":"External Pointers","kind":"list","data": snippets},
            ],
            "narrative": _coach_narrative(analysis["summary"], analysis["funnel"], cards[:3]),
        }
        return ui
    except Exception as e:
        _set_error(e)
        return JSONResponse(status_code=500, content={"detail":"Coach failed. See /debug/last_error."})

# ---------------------------
# Prompt Lab
# ---------------------------
class PromptLabIn(BaseModel):
    goal: str
    context: str = ""

@app.post("/prompt_lab")
def prompt_lab(inp: PromptLabIn):
    try:
        # Pick a template based on the goal
        g = (inp.goal or "").lower()
        if any(k in g for k in ["vsl","video","script"]):
            template = "vsl_depth"
            prompt = (
                "You are a senior DTC creative strategist.\n"
                f"Goal: {inp.goal}\n"
                f"Audience/context: {inp.context}\n\n"
                "Output a VSL outline with:\n"
                "1) Hook options (10, ≤8 words each, pattern-break)\n"
                "2) Problem agitation (3 bullets)\n"
                "3) Product mechanism explained simply (ATP/mitochondria if applicable)\n"
                "4) Social proof (3 micro-testimonials)\n"
                "5) Offer framing (value stack + price anchor)\n"
                "6) CTA lines (5 variants)\n"
                "7) Compliance-sounding safety language (e.g., ‘individual results vary’)\n"
                "Format in clean bullets only."
            )
        elif any(k in g for k in ["ugc","hook","ad"]):
            template = "ugc_hooks"
            prompt = (
                "You are a DTC creative strategist.\n"
                f"Goal: {inp.goal}\n"
                f"Audience/context: {inp.context}\n\n"
                "Generate 20 UGC hooks (≤8 words) that stop scroll using pain, curiosity, proof angles.\n"
                "Return a simple numbered list."
            )
        else:
            template = "structured_brief"
            prompt = (
                "You are a performance creative brief generator.\n"
                f"Goal: {inp.goal}\n"
                f"Audience/context: {inp.context}\n\n"
                "Return: key insight, big idea, 3 angles, 3 thumbnails, opening lines, CTA lines, and a 30s script."
            )
        return {"goal": inp.goal, "template_used": template, "prompt": prompt}
    except Exception as e:
        _set_error(e)
        return JSONResponse(status_code=500, content={"detail":"Prompt Lab failed. See /debug/last_error."})

# ---------------------------
# Script Doctor
# ---------------------------
class ScriptIn(BaseModel):
    script: str
    product: str = ""
    audience: str = ""

@app.post("/script_doctor")
def script_doctor(inp: ScriptIn):
    try:
        text = (inp.script or "").strip()
        segs = []
        notes = []

        # Lightweight segmentation (heuristic)
        t_low = text.lower()
        if any(s in t_low for s in ["how", "what", "why", "are you"]):
            segs.append("Hook")
        if any(s in t_low for s in ["pain", "struggle", "tired", "problem"]):
            segs.append("Problem/Agitate")
        if any(s in t_low for s in ["because", "it's", "it is", "works by", "uses"]):
            segs.append("Mechanism")
        if any(s in t_low for s in ["unlike", "instead", "different"]):
            segs.append("Differentiate")
        if any(s in t_low for s in ["just wrap", "easy", "simple", "no pain"]):
            segs.append("Ease-of-use")
        if any(s in t_low for s in ["testimonial", "reviews", "before", "after", "thousand"]):
            segs.append("Proof")
        if any(s in t_low for s in ["now", "today", "limited", "act"]):
            segs.append("CTA")

        # Recommendations inspired by Shaun/Spencer patterns
        notes.append("Add a crisp 1-line hook with a hard pattern break in the first 2s.")
        notes.append("Insert quick proof (testimonial/number/before-after) in 8–12s window.")
        notes.append("Name the mechanism in plain language, then show it (not just tell).")
        notes.append("Close with a direct CTA + a reason to act now (micro-urgency).")
        notes.append("If hooks/holds are low in account, try a ‘show the outcome first’ cold open.")

        return {"segments_detected": list(dict.fromkeys(segs)) or ["(Could not confidently segment)"], "notes": notes}
    except Exception as e:
        _set_error(e)
        return JSONResponse(status_code=500, content={"detail":"Script Doctor failed. See /debug/last_error."})

# ---------------------------
# Debug
# ---------------------------
@app.get("/debug/last_error")
def last_error():
    return {"last_error": MEMORY["last_error"]}

# ============================================================
# Minimal Dark UI
# ============================================================

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Erto Agent</title>
  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
  <style>
    body { background: #0b1220; color: #e5e7eb; }
    .card { background: rgba(17,24,39,0.7); border: 1px solid #1f2937; }
    .pill { background: rgba(37,99,235,0.2); border: 1px solid rgba(59,130,246,0.4); }
    .muted { color:#93a3b8; }
    code { background:#0f172a; padding:2px 6px; border-radius:6px; }
  </style>
</head>
<body class="p-6">
  <div class="max-w-5xl mx-auto space-y-6">
    <h1 class="text-2xl font-semibold">Erto Agent</h1>
    <p class="muted">Upload your Meta CSV, then open <code>/docs</code> or call the endpoints below.</p>

    <div class="grid md:grid-cols-3 gap-4">
      <div class="card rounded-xl p-4">
        <div class="font-medium">1) Ingest CSV</div>
        <div class="muted text-sm mt-1">POST <code>/ingest_csv</code> (multipart file)</div>
        <div class="mt-2 text-sm">Use Swagger: <a class="text-blue-400 underline" href="/docs">/docs</a></div>
      </div>
      <div class="card rounded-xl p-4">
        <div class="font-medium">2) Coach</div>
        <div class="muted text-sm mt-1">GET <code>/coach</code> → analysis, decisions, playbook</div>
      </div>
      <div class="card rounded-xl p-4">
        <div class="font-medium">3) Prompt Lab</div>
        <div class="muted text-sm mt-1">POST <code>/prompt_lab</code> (goal, context)</div>
      </div>
      <div class="card rounded-xl p-4">
        <div class="font-medium">4) Script Doctor</div>
        <div class="muted text-sm mt-1">POST <code>/script_doctor</code> (script)</div>
      </div>
      <div class="card rounded-xl p-4">
        <div class="font-medium">Debug</div>
        <div class="muted text-sm mt-1">GET <code>/debug/last_error</code></div>
      </div>
    </div>

    <div class="card rounded-xl p-4">
      <div class="font-medium">Optional: External Suggestions</div>
      <div class="muted text-sm mt-1">Set <code>SERPER_API_KEY</code> env and flip <code>SETTINGS["external_data"]=True</code> in the code to show quick outside ideas aligned to your biggest funnel leak.</div>
    </div>
  </div>
</body>
</html>
"""

# ============================================================
# Done.
# ============================================================
