import os
import io
import math
import json
import traceback
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import httpx

# -------------------------------
# Global state (simple, file-backed)
# -------------------------------
DATA_PATH = "/tmp/ad_metrics.parquet"
LAST_ERROR: str = ""
DATA_DF: Optional[pd.DataFrame] = None

SETTINGS = {
    "breakeven_roas": 1.54,
    "target_roas": 2.0,
    "min_test_spend": 20.0,
    "min_scale_spend": 50.0,
    "max_cpr": None,  # optional kill threshold if provided
    # soft metric “good” bars (fractions, not %)
    "hook_good": 0.30,
    "hold_good": 0.10,
    "ctr_good": 0.015,
    # external tips (SerpAPI) – used ONLY for tips/suggestions, not metrics
    "external_tips_enabled": True,
    "external_tips_max": 4,
}

SERP_API_KEY = os.getenv("SERP_API_KEY", "").strip()

# -------------------------------
# App
# -------------------------------
app = FastAPI(title="Erto Agent", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

# -------------------------------
# Utils
# -------------------------------

def set_last_error(msg: str) -> None:
    global LAST_ERROR
    LAST_ERROR = msg


def load_data_from_disk() -> Optional[pd.DataFrame]:
    if os.path.exists(DATA_PATH):
        try:
            return pd.read_parquet(DATA_PATH)
        except Exception as e:
            set_last_error(f"Failed to read parquet: {e}")
    return None


def save_data_to_disk(df: pd.DataFrame) -> None:
    try:
        df.to_parquet(DATA_PATH, index=False)
    except Exception as e:
        set_last_error(f"Failed to write parquet: {e}")


def to_float(x) -> float:
    try:
        if pd.isna(x):
            return 0.0
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        v = float(x)
        return v
    except Exception:
        return 0.0


def frac_from_maybe_percent(v: float) -> float:
    # normalize: if someone fed 12.5 meaning "12.5%" → 0.125
    if pd.isna(v):
        return 0.0
    v = float(v)
    if v > 1.0:
        return v / 100.0
    return v


def pick_first_numeric(df: pd.DataFrame, names: List[str]) -> str:
    for n in names:
        if n in df.columns:
            return n
    return ""


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw Meta CSV columns to a normalized schema.

    Normalized columns:
      ad_name, ad_set, campaign, day, spend, clicks, purchases, roas, cpr, ctr, hook, hold,
      add_to_cart, initiate_checkout, reach, freq, cpm, video_play, gender, age
    """
    colmap: Dict[str, str] = {}

    # text id/name fields
    if "Ad name" in df.columns:
        colmap["Ad name"] = "ad_name"
    elif "Ad" in df.columns:
        colmap["Ad"] = "ad_name"

    if "Ad set name" in df.columns:
        colmap["Ad set name"] = "ad_set"
    if "Campaign name" in df.columns:
        colmap["Campaign name"] = "campaign"
    if "Day" in df.columns:
        colmap["Day"] = "day"
    if "Gender" in df.columns:
        colmap["Gender"] = "gender"
    if "Age" in df.columns:
        colmap["Age"] = "age"

    # spend
    spend_col = pick_first_numeric(df, ["Amount spent (USD)", "Amount spent", "Spend", "amount_spent"])
    if spend_col:
        colmap[spend_col] = "spend"

    # clicks
    clicks_col = pick_first_numeric(df, ["Link clicks", "Clicks (all)", "link_clicks", "Clicks"])
    if clicks_col:
        colmap[clicks_col] = "clicks"

    # purchases (prefer Website purchases if present)
    purch_col = pick_first_numeric(df, ["Website purchases", "Purchases", "purchases", "Total purchases"])
    if purch_col:
        colmap[purch_col] = "purchases"

    # ROAS
    roas_col = pick_first_numeric(df, [
        "Website purchase ROAS (return on advertising spend)",
        "Purchase ROAS (return on ad spend)",
        "ROAS",
        "purchase_roas",
    ])
    if roas_col:
        colmap[roas_col] = "roas"

    # CPR / CPA (Meta sometimes calls it "Cost per purchase" or "Cost per result")
    cpr_col = pick_first_numeric(df, ["Cost per purchase", "Cost per result", "CPA", "cpr"])
    if cpr_col:
        colmap[cpr_col] = "cpr"

    # CTR
    ctr_col = pick_first_numeric(df, ["CTR (link click-through rate)", "CTR", "link_ctr"])
    if ctr_col:
        colmap[ctr_col] = "ctr"

    # Video metrics we’ll treat as “hook” & “hold” from custom metrics if present
    hook_col = pick_first_numeric(df, ["hook rate", "Video hook", "Hook"])
    if hook_col:
        colmap[hook_col] = "hook"

    hold_col = pick_first_numeric(df, ["hold rate", "Video hold", "Hold"])
    if hold_col:
        colmap[hold_col] = "hold"

    # funnel
    atc_col = pick_first_numeric(df, ["Website adds to cart", "Adds to cart", "Add to cart"])
    if atc_col:
        colmap[atc_col] = "add_to_cart"
    ic_col = pick_first_numeric(df, ["Website checkouts initiated", "Checkouts initiated"])
    if ic_col:
        colmap[ic_col] = "initiate_checkout"

    # reach, frequency, cpm
    if "Reach" in df.columns:
        colmap["Reach"] = "reach"
    if "Frequency" in df.columns:
        colmap["Frequency"] = "freq"
    cpm_col = pick_first_numeric(df, ["CPM (cost per 1,000 impressions)", "CPM"])
    if cpm_col:
        colmap[cpm_col] = "cpm"

    # apply rename
    ndf = df.rename(columns=colmap).copy()

    # to numerics
    for num in [
        "spend", "clicks", "purchases", "roas", "cpr",
        "ctr", "hook", "hold", "add_to_cart", "initiate_checkout",
        "reach", "freq", "cpm",
    ]:
        if num in ndf.columns:
            ndf[num] = pd.to_numeric(ndf[num], errors="coerce")

    # normalize fractions
    for frac in ["ctr", "hook", "hold"]:
        if frac in ndf.columns:
            ndf[frac] = ndf[frac].apply(frac_from_maybe_percent)

    # fill required columns
    for col in ["ad_name", "spend", "clicks", "purchases", "roas", "cpr", "ctr", "hook", "hold", "add_to_cart", "initiate_checkout"]:
        if col not in ndf.columns:
            ndf[col] = 0

    # clean ad_name
    ndf["ad_name"] = ndf["ad_name"].fillna("Unknown Ad").astype(str)

    return ndf


def summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    spend = float(df["spend"].sum())
    purchases = int(df["purchases"].sum())
    clicks = int(df["clicks"].sum())
    # avg roas: if present, prefer revenue/spend; else weighted roas
    if "roas" in df.columns and df["roas"].notna().any():
        # if we had purchase conversion value we’d do revenue/spend;
        # here we approximate weighted mean by spend
        w = df["spend"].replace(0, pd.NA)
        avg_roas = float((df["roas"] * df["spend"]).sum() / max(spend, 1e-9))
    else:
        avg_roas = 0.0
    return {
        "spend": round(spend, 2),
        "purchases": purchases,
        "clicks": clicks,
        "avg_roas": round(avg_roas, 2),
        "settings": SETTINGS,
    }


def blended_funnel(df: pd.DataFrame) -> Tuple[Dict[str, Any], List[str]]:
    clicks = int(df["clicks"].sum()) if "clicks" in df else 0
    atc = int(df["add_to_cart"].sum()) if "add_to_cart" in df else 0
    ic = int(df["initiate_checkout"].sum()) if "initiate_checkout" in df else 0
    p = int(df["purchases"].sum()) if "purchases" in df else 0

    def safe_div(a, b): return (a / b) if b else 0.0
    diag: List[str] = []

    c2a = safe_div(atc, clicks)
    a2i = safe_div(ic, max(atc, 1))
    i2p = safe_div(p, max(ic, 1))

    # simple heuristics for leaks
    if i2p < 0.35:
        diag.append("Low Checkout→Purchase (checkout UX friction, payment issues, reassurance missing).")
    if c2a < 0.20:
        diag.append("Low Click→ATC (offer/page mis-match, weak trust, slow LP).")
    if a2i < 0.35:
        diag.append("Low ATC→IC (cart friction, shipping surprises, weak urgency).")

    return ({
        "clicks": clicks,
        "add_to_cart": atc,
        "initiate_checkout": ic,
        "purchases": p,
        "click_to_atc": round(c2a, 4),
        "atc_to_ic": round(a2i, 4),
        "ic_to_purchase": round(i2p, 4),
    }, diag)


def score_row(r: pd.Series) -> float:
    """
    Lightweight composite score: favors ROAS + signal (hook/hold/ctr) + sufficient spend.
    """
    roas = to_float(r.get("roas"))
    spend = to_float(r.get("spend"))
    hook = to_float(r.get("hook"))
    hold = to_float(r.get("hold"))
    ctr = to_float(r.get("ctr"))
    purchases = to_float(r.get("purchases"))

    # normalize signals to 0..1 against “good bars”
    s_hook = min(hook / max(SETTINGS["hook_good"], 1e-9), 2.0)
    s_hold = min(hold / max(SETTINGS["hold_good"], 1e-9), 2.0)
    s_ctr = min(ctr / max(SETTINGS["ctr_good"], 1e-9), 2.0)
    s_signal = (s_hook + s_hold + s_ctr) / 3.0

    s_spend = math.tanh(spend / max(SETTINGS["min_test_spend"], 1e-9))  # 0..1
    s_conv = math.tanh(purchases / 2.0)  # small push if it has some purchases

    score = (0.55 * roas) + (0.25 * s_signal) + (0.15 * s_spend) + (0.05 * s_conv)
    return float(round(score, 3))


def tips_for_row(r: pd.Series) -> List[str]:
    t: List[str] = []
    roas = to_float(r.get("roas"))
    spend = to_float(r.get("spend"))
    hook = to_float(r.get("hook"))
    hold = to_float(r.get("hold"))
    ctr = to_float(r.get("ctr"))
    cpr = to_float(r.get("cpr"))
    purchases = to_float(r.get("purchases"))

    # normalized checks (good bars)
    good_hook = hook >= SETTINGS["hook_good"]
    good_hold = hold >= SETTINGS["hold_good"]
    good_ctr = ctr >= SETTINGS["ctr_good"]

    if spend < SETTINGS["min_test_spend"]:
        t.append("Not enough spend yet—let it gather data ≥ test threshold.")
    if roas >= SETTINGS["target_roas"]:
        t.append("On track—watch for 72h stability (≥60% 7d-click) then scale +20%.")
    elif roas < SETTINGS["breakeven_roas"] and spend >= SETTINGS["min_test_spend"]:
        t.append("Below breakeven—pause or iterate creative and relaunch.")
    if SETTINGS["max_cpr"] and cpr and cpr > SETTINGS["max_cpr"]:
        t.append(f"CPR above threshold (${SETTINGS['max_cpr']:.2f})—kill or overhaul.")

    # creative signal guidance
    if not good_hook:
        t.append("Hook under benchmark—test a 2–3s pattern-breaker opener (motion + claim).")
    if good_hook and not good_hold:
        t.append("Strong hook but weak hold—tighten first 5–12s with curiosity ladder.")
    if not good_ctr:
        t.append("Low CTR—tighten headline/thumbnail; add reason-to-click in first line.")

    if roas == 0 and purchases == 0 and spend >= SETTINGS["min_test_spend"]:
        t.append("No conversions at test spend—diagnose landing page match & offer.")

    return t[:4] if t else ["Monitor for another 24h; no decisive signal yet."]


def bucketize(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for _, r in df.groupby("ad_name").agg({
        "spend": "sum",
        "purchases": "sum",
        "roas": "mean",
        "cpr": "mean",
        "ctr": "mean",
        "hook": "mean",
        "hold": "mean"
    }).reset_index().iterrows():
        d = dict(r)
        d["cvr"] = (to_float(r["purchases"]) / max(1.0, to_float(df[df["ad_name"] == r["ad_name"]]["clicks"].sum())))
        d["score"] = score_row(pd.Series(d))
        d["tips"] = tips_for_row(pd.Series(d))
        rows.append({
            "ad_id": r["ad_name"],
            "name": r["ad_name"],
            "spend": round(to_float(r["spend"]), 2),
            "roas": round(to_float(r["roas"]), 2),
            "cpr": round(to_float(r["cpr"]), 2),
            "cvr": round(d["cvr"], 4),
            "ctr": round(to_float(r["ctr"]), 4),
            "hook": round(to_float(r["hook"]), 4),
            "hold": round(to_float(r["hold"]), 4),
            "score": d["score"],
            "tips": d["tips"],
        })

    # sort by score
    rows.sort(key=lambda x: x["score"], reverse=True)

    scale, iterate, kill = [], [], []
    for a in rows:
        if a["spend"] >= SETTINGS["min_scale_spend"] and a["roas"] >= SETTINGS["target_roas"]:
            scale.append(a)
        elif a["spend"] < SETTINGS["min_test_spend"]:
            iterate.append(a)
        elif a["roas"] < SETTINGS["breakeven_roas"]:
            kill.append(a)
        else:
            # neither → keep in iterate (optimize)
            iterate.append(a)

    return {"scale": scale[:12], "iterate": iterate[:20], "kill": kill[:12], "all_ranked": rows}


async def external_tips(query: str, limit: int = 4) -> List[Dict[str, str]]:
    """
    Best-effort SerpAPI pull, used ONLY for enriching tips/suggestions (never metrics).
    """
    if not SETTINGS["external_tips_enabled"] or not SERP_API_KEY:
        return []
    try:
        params = {
            "engine": "google",
            "q": query,
            "hl": "en",
            "gl": "us",
            "api_key": SERP_API_KEY,
            "num": max(5, limit + 2),
        }
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.get("https://serpapi.com/search.json", params=params)
            r.raise_for_status()
            data = r.json()
        outs = []
        for it in data.get("organic_results", [])[:limit]:
            outs.append({
                "title": it.get("title", "")[:140],
                "snippet": (it.get("snippet") or it.get("rich_snippet", {}).get("top", {}).get("extensions", [""])[0] if isinstance(it.get("rich_snippet", {}).get("top", {}).get("extensions", []), list) else "")[:220]
            })
        return outs
    except Exception as e:
        set_last_error(f"SerpAPI error: {e}")
        return []


def shaun_spencer_playbook(funnel_diag: List[str]) -> List[str]:
    """More descriptive, course-driven guidance (reasoned)."""
    g: List[str] = []
    g.append("Check attribution split: aim ≥60% purchases from 7d-click before scaling; if heavy 1d-view, expect volatility.")
    if any("Checkout→Purchase" in x for x in funnel_diag):
        g.append("Checkout friction: verify payment methods, trust badges near pay button, shipping calculator upfront, and micro-urgency (limited stock, 24h bonus).")
    if any("Click→ATC" in x for x in funnel_diag):
        g.append("Weak click→ATC: tighten LP hierarchy (benefit headline, 3-bullet proof, social proof above fold), reduce first paint >2.5s.")
    if any("ATC→IC" in x for x in funnel_diag):
        g.append("Cart drop: remove surprise costs, default cheapest shipping, surface returns policy, and add sticky ‘Continue’ CTA.")
    g.append("Scaling protocol: ≥72h above target KPI → +20% budget; if <72h, wait 24h. If below breakeven at test spend, rotate new DCTs.")
    return g


def potential_winners(rows: List[Dict[str, Any]]) -> List[str]:
    names = []
    for r in rows[:6]:
        # “watching” if decent score + not yet scale bucket
        if r["score"] >= 1.3 and r["spend"] >= (0.5 * SETTINGS["min_test_spend"]):
            names.append(r["name"])
    return names[:6]


def coach_answer(question: str, summ: Dict[str, Any], funnel: Dict[str, Any], diag: List[str]) -> str:
    q = (question or "").lower()
    spend = summ["spend"]
    p = summ["purchases"]
    roas = summ["avg_roas"]

    if "roas" in q:
        return f"Your blended ROAS is {roas:.2f} on ${spend:.2f} spend ({p} purchases). Target is {SETTINGS['target_roas']:.2f}."

    if "scale" in q or "budget" in q:
        return ("Scale if ≥72h above target and (ideally) ≥60% 7d-click. If so, increase +20%. "
                "If below breakeven at ≥ test spend, hold budgets and rotate new creatives.")

    if "dropoff" in q or "why" in q:
        parts = []
        parts.append(f"Click→ATC {funnel['click_to_atc']*100:.1f}%, ATC→IC {funnel['atc_to_ic']*100:.1f}%, IC→Purchase {funnel['ic_to_purchase']*100:.1f}%.")
        if diag:
            parts.append("Likely causes: " + "; ".join(diag))
        parts.append("Address the biggest leak first (one change at a time) and let it run 24–48h.")
        return " ".join(parts)

    # default helpful
    return ("Ask me things like: 'What’s my ROAS?', 'Should I scale today?', "
            "'Why is Click→ATC low?', or 'Give me 3 actions for the next 24h'.")


def script_doctor_segments(text: str) -> Dict[str, Any]:
    s = text.strip()
    segs = []
    notes = []

    # Very lightweight tagging
    if len(s) < 12:
        notes.append("Script is too short—aim 120–300 words for a short UGC.")
    # detect pieces
    if "?" in s.split("\n")[0][:100]:
        segs.append("Hook")
    if "because" in s.lower() or "it’s" in s.lower():
        segs.append("Problem / Mechanism")
    if "unlike" in s.lower() or "proof" in s.lower() or "clinical" in s.lower():
        segs.append("Differentiation / Proof")
    if "feel" in s.lower():
        segs.append("Emotion / Benefit")
    if "wrap" in s.lower() or "how to" in s.lower():
        segs.append("Demo")
    if "confident" in s.lower() or "back" in s.lower():
        segs.append("Transformation")
    if "now" in s.lower() or "today" in s.lower() or "shop" in s.lower():
        segs.append("CTA")

    # course-flavored fixes
    notes.append("Add a crisp 1-line hook in the first 2s with a hard pattern break.")
    notes.append("Inject a quick proof bit within 8–12s (testimonial, number, before/after).")
    notes.append("Close with direct CTA + micro-urgency (bonus, limited stock, timer).")

    return {
        "segments_detected": list(dict.fromkeys(segs)) or ["Unknown"],
        "notes": notes[:6]
    }

# -------------------------------
# Routes
# -------------------------------

@app.get("/health")
def health():
    return {"ok": True, "has_data": DATA_DF is not None}

@app.get("/debug/last_error")
def last_error():
    return {"last_error": LAST_ERROR or "—"}

@app.post("/ingest_csv")
async def ingest_csv(file: UploadFile = File(...)):
    """
    Upload any Meta CSV (any date range). Rows append into normalized in-memory store.
    """
    global DATA_DF
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
        ndf = normalize_df(df)

        existing = load_data_from_disk()
        if existing is not None:
            DATA_DF = pd.concat([existing, ndf], ignore_index=True)
        else:
            DATA_DF = ndf

        save_data_to_disk(DATA_DF)

        return {"ok": True, "rows": int(len(ndf)), "columns": list(ndf.columns)}
    except Exception:
        err = traceback.format_exc(limit=2)
        set_last_error(err)
        return JSONResponse({"detail": "Server failed to ingest CSV. Hit /debug/last_error for details."}, status_code=500)

@app.post("/ingest_csv_debug", response_model=None)
async def ingest_csv_debug(file: UploadFile = File(...)):
    """
    Parse-only: show detected columns, dtypes, and first 5 rows (no DB write).
    """
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
        ndf = normalize_df(df)
        return {
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "sample": ndf.head(5).replace({pd.NA: None}).to_dict(orient="records"),
        }
    except Exception:
        err = traceback.format_exc(limit=2)
        set_last_error(err)
        return JSONResponse({"detail": "Parse failed. See /debug/last_error."}, status_code=500)

@app.get("/analyze")
async def analyze():
    global DATA_DF
    try:
        if DATA_DF is None:
            DATA_DF = load_data_from_disk()
        if DATA_DF is None or DATA_DF.empty:
            return JSONResponse({"detail": "No data ingested yet."}, status_code=400)

        df = DATA_DF.copy()
        summ = summary_stats(df)
        funnel, diag = blended_funnel(df)
        buckets = bucketize(df)

        watching = potential_winners(buckets["all_ranked"])

        guidance = shaun_spencer_playbook(diag)

        # external nuggets ONLY for guidance enrichment
        ext_snips: List[Dict[str, str]] = []
        if SETTINGS["external_tips_enabled"] and SERP_API_KEY:
            # Build a query reflecting biggest leak or scaling needs
            q = ""
            if any("Checkout→Purchase" in d for d in diag):
                q = "reduce checkout abandonment ecommerce best practices trust payment reassurance"
            elif any("Click→ATC" in d for d in diag):
                q = "improve product page conversion rate ecommerce above the fold trust social proof"
            elif any("ATC→IC" in d for d in diag):
                q = "increase add to cart to checkout rate reduce cart abandonment ecommerce"
            else:
                q = "facebook ads scaling protocol 7 day click attribution best practices"
            ext_snips = await external_tips(q, SETTINGS["external_tips_max"])

        payload = {
            "summary": summ,
            "funnel": {"blended": funnel, "diagnosis": diag},
            "watching": watching,
            "buckets": {k: v for k, v in buckets.items() if k != "all_ranked"},
            "guidance": guidance,
            "external_snippets": ext_snips,
        }
        return JSONResponse(payload)
    except Exception:
        err = traceback.format_exc(limit=2)
        set_last_error(err)
        return JSONResponse({"detail": "Analyze failed. See /debug/last_error."}, status_code=500)

@app.post("/coach")
async def coach(body: Dict[str, Any] = Body(...)):
    """
    Lightweight Q&A that references current data; will optionally enrich *reasons/tips* with external nuggets.
    """
    global DATA_DF
    try:
        q = str(body.get("question", "")).strip()
        if DATA_DF is None:
            DATA_DF = load_data_from_disk()
        if DATA_DF is None or DATA_DF.empty:
            return JSONResponse({"detail": "No data ingested yet."}, status_code=400)

        df = DATA_DF.copy()
        summ = summary_stats(df)
        funnel, diag = blended_funnel(df)
        answer = coach_answer(q, summ, funnel, diag)

        ext: List[Dict[str, str]] = []
        if SETTINGS["external_tips_enabled"] and SERP_API_KEY:
            # Only enrich if user seems to want advice or “why”
            if any(k in q.lower() for k in ["tips", "suggest", "improve", "why", "drop", "diagnose", "scale"]):
                q2 = "facebook ads diagnose funnel dropoff ecommerce" if "drop" in q.lower() or "why" in q.lower() else "facebook ads scaling 7 day click attribution"
                ext = await external_tips(q2, 3)

        return JSONResponse({
            "summary": summ,
            "funnel": {"blended": funnel, "diagnosis": diag},
            "answer": answer,
            "external_snippets": ext
        })
    except Exception:
        err = traceback.format_exc(limit=2)
        set_last_error(err)
        return JSONResponse({"detail": "Coach failed. See /debug/last_error."}, status_code=500)

@app.post("/prompt_lab")
async def prompt_lab(body: Dict[str, Any] = Body(...)):
    """
    Turn Goal + Context into a strong prompt. Returns a template + final prompt.
    """
    goal = str(body.get("goal", "")).strip()
    context = str(body.get("context", "")).strip()

    if not goal:
        goal = "Generate 10 UGC hooks for a VSL."
    template = "VSL_builder"

    prompt = (
        "You are a senior DTC creative strategist.\n"
        f"Goal: {goal}\n"
        f"Audience/Context: {context or 'N/A'}\n\n"
        "Output:\n"
        "1) 10 scroll-stopping hooks (≤8 words) mixing pain, curiosity, proof.\n"
        "2) 3 core angles with promise + proof.\n"
        "3) 60–90s VSL outline: Hook → Problem → Mechanism → Proof → Offer → CTA.\n"
        "4) 3 alt CTAs with micro-urgency.\n"
        "Style: concise bullets, no fluff."
    )

    return {"goal": goal, "template_used": template, "prompt": prompt}

@app.post("/script_doctor")
async def script_doctor(body: Dict[str, Any] = Body(...)):
    """
    Detect sections and return specific, actionable improvements.
    """
    script = str(body.get("script", "")).strip()
    if not script:
        return JSONResponse({"detail": "No script provided."}, status_code=400)

    out = script_doctor_segments(script)
    # extra deep suggestions
    suggestions = [
        "Tighten first 5–12s: state the promise in plain language, then tease the mechanism (“why it works”).",
        "Insert 1 concrete proof: named testimonial, specific number, or before/after claim (compliant).",
        "Land the offer with value stack (what you get) before price; then a single risk-reversal line.",
        "CTA should specify the next click and why now (limited bonus or trial window)."
    ]
    out["suggestions"] = suggestions
    return out

# -------------------------------
# UI
# -------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Erto Agent</title>
<style>
  :root { --bg:#0b1217; --panel:#0f1620; --card:#111a22; --edge:#1e293b; --muted:#8aa0b6; --text:#e2e8f0; --accent:#22d3ee;}
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 800px at 10% -10%, #0e1a22 0%, #0b1217 55%, #091017 100%);color:var(--text);font:14px/1.45 system-ui, -apple-system, Segoe UI, Roboto, Inter, sans-serif}
  .wrap{max-width:1180px;margin:28px auto;padding:0 16px}
  .tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
  .tab{padding:10px 14px;border-radius:12px;background:var(--panel);border:1px solid var(--edge);cursor:pointer;transition:.15s}
  .tab:hover{filter:brightness(1.08)}
  .tab.active{background:var(--card);border-color:#2a3b50;box-shadow:0 0 0 1px rgba(255,255,255,.03) inset}
  .card{background:var(--card);border:1px solid var(--edge);border-radius:14px;padding:16px;margin-bottom:16px;box-shadow:0 0 0 1px rgba(255,255,255,.02) inset}
  h3{margin:0 0 10px 0}
  .muted{color:var(--muted)}
  textarea,input,button{width:100%;border-radius:10px;border:1px solid #243243;background:#0b141c;color:var(--text);padding:10px}
  button{cursor:pointer;background:#0c3a45;border-color:#0f4a57}
  code{background:#0f1822;padding:2px 6px;border-radius:6px;border:1px solid #223041}
  .grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
  .grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  @media(max-width:980px){ .grid4{grid-template-columns:repeat(2,1fr)} .grid3{grid-template-columns:1fr} .grid2{grid-template-columns:1fr}}
  .pill{display:inline-block;padding:2px 10px;border-radius:999px;border:1px solid #284050;background:#0b151d;margin:2px 6px 6px 0}
  .bubble{display:inline-block;background:#0f1822;border:1px solid #223244;padding:6px 10px;border-radius:999px;margin:4px 6px 0 0}
  .ad{border:1px solid #213141;border-radius:12px;padding:12px;background:#0e1720}
  .ad h4{margin:0 0 8px 0;font-weight:600}
  .small{font-size:12px}
  .section-title{margin-bottom:8px;font-weight:600;letter-spacing:.2px}
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
    <div class="grid2">
      <input id="p_goal" placeholder="Goal (e.g., generate a VSL outline)"/>
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
  // tabs
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

  function pill(k,v){ return `<span class="pill">${k}: <b>${v}</b></span>`; }
  function section(title, html){ return `<div class="card"><div class="section-title">${title}</div>${html}</div>`; }

  function renderAds(arr){
    if(!arr || !arr.length) return '<div class="muted">—</div>';
    return arr.map(a=>`
      <div class="ad">
        <h4>${a.name || a.ad_id}</h4>
        <div class="muted small">Score ${a.score} • Spend $${a.spend} • ROAS ${a.roas} • CPR $${a.cpr} • CVR ${(a.cvr*100).toFixed(1)}% • CTR ${(a.ctr*100).toFixed(1)}% • Hook ${(a.hook*100).toFixed(1)}% • Hold ${(a.hold*100).toFixed(1)}%</div>
        <div style="margin-top:6px">${(a.tips||[]).map(t=>`<span class="bubble">${t}</span>`).join('')}</div>
      </div>`).join('');
  }

  async function runAnalyze(){
    const r = await fetch('/analyze');
    if(!r.ok){ document.getElementById('an_out').innerHTML = '<div class="muted">Upload a CSV first.</div>'; return; }
    const d = await r.json();

    let kpis = `<div class="grid4">
      <div class="card"><div class="muted small">Spend</div><div><b>$${d.summary.spend}</b></div></div>
      <div class="card"><div class="muted small">Purchases</div><div><b>${d.summary.purchases}</b></div></div>
      <div class="card"><div class="muted small">Avg ROAS</div><div><b>${d.summary.avg_roas}</b></div></div>
      <div class="card"><div class="muted small">Target ROAS</div><div><b>${d.summary.settings.target_roas}</b></div></div>
    </div>`;

    const f = d.funnel.blended;
    let funnel = pill('Click→ATC', (f.click_to_atc*100).toFixed(1)+'%')+' '+
                 pill('ATC→IC', (f.atc_to_ic*100).toFixed(1)+'%')+' '+
                 pill('IC→Purchase', (f.ic_to_purchase*100).toFixed(1)+'%');

    let diag = (d.funnel.diagnosis||[]).map(x=>`<span class="bubble">${x}</span>`).join('');
    let watch = (d.watching||[]).map(w=>`<span class="bubble">${w}</span>`).join('') || '<span class="muted">—</span>';

    let buckets = `
      <div class="grid3">
        <div>${section('Scale', renderAds(d.buckets.scale))}</div>
        <div>${section('Iterate', renderAds(d.buckets.iterate))}</div>
        <div>${section('Kill', renderAds(d.buckets.kill))}</div>
      </div>`;

    let guide = (d.guidance||[]).map(g=>`<span class="bubble">${g}</span>`).join('');

    let ext = '';
    if(d.external_snippets && d.external_snippets.length){
      ext += '<div class="card"><div class="section-title">External nuggets (tips only)</div>';
      d.external_snippets.forEach(s=> ext += `<div class="small">• <b>${s.title}</b> — ${s.snippet}</div>`);
      ext += '</div>';
    }

    document.getElementById('an_out').innerHTML =
      section('Summary', kpis) +
      section('Funnel', funnel + '<div style="margin-top:8px">'+diag+'</div>') +
      section('Watching (potential winners)', watch) +
      buckets +
      section('Playbook', guide) +
      ext;
  }

  async function askCoach(){
    const q = document.getElementById('coach_q').value || 'What should I do next?';
    const r = await fetch('/coach', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question:q})});
    const d = await r.json();
    let html = `<div class="card"><div>${d.answer}</div></div>`;
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

# done
