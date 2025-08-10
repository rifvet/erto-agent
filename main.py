# main.py
# ERTO Agent — lightweight, DB-free version
# FastAPI app with: CSV ingest, Coach, Prompt Lab, Script Doctor, and a clean dark UI.

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import traceback
import datetime as dt

app = FastAPI(title="ERTO Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def _to_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    try:
        s = str(x).strip().replace(",", "")
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        val = float(s)
        return val
    except Exception:
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
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "-"

GOOD = {"hook": SETTINGS["hook_good"], "hold": SETTINGS["hold_good"], "ctr": SETTINGS["ctr_good"]}

def _growth_score(row):
    roas   = float(row.get("roas", 0) or 0)
    cpr    = float(row.get("cpr", 0) or 0)
    ctr    = float(row.get("ctr", 0) or 0)
    hook   = float(row.get("hook", 0) or 0)
    hold   = float(row.get("hold", 0) or 0)
    spend  = float(row.get("spend", 0) or 0)

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
    return {
        "title": row.get("name") or row.get("ad") or "Ad",
        "score": _growth_score(row),
        "metrics": metrics,
        "tips": tips,
        "raw": row,
    }

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

        # append if we already have data
        if LAST_DF is not None:
            LAST_DF = pd.concat([LAST_DF, norm], ignore_index=True)
        else:
            LAST_DF = norm

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

    # aggregate per ad
    agg = g.agg({
        "spend": "sum",
        "revenue": "sum",
        "purchases": "sum",
        "clicks": "sum",
        "ctr": "mean",
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

    diagnosis = []
    if funnel_blended["ic_to_purchase"] < 0.4 and purchases >= 5:
        diagnosis.append("Low Checkout→Purchase (checkout friction, payment issues, missing reassurance).")
    if funnel_blended["click_to_atc"] < 0.2 and clicks >= 200:
        diagnosis.append("Low Click→ATC (offer clarity, price anchoring, trust signals).")
    if not diagnosis:
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

    decisions = {"scale": SCALE, "iterate": ITERATE, "kill": KILL}

    return {"summary": summary, "funnel": {"blended": funnel_blended, "diagnosis": diagnosis}, "decisions": decisions}

@app.post("/coach")
def coach():
    try:
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
    except Exception as e:
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
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ERTO Agent</title>
<script src="https://cdn.tailwindcss.com"></script>
</head>
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
  </section>
</div>

<script>
const tabs = document.querySelectorAll('.tab');
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
function renderCoach(ui){
  document.getElementById('coach').classList.remove('hidden');
  document.querySelector('[data-tab="coach"]').classList.replace('bg-slate-800','bg-slate-700');
  document.getElementById('coach-narrative').innerHTML = `<div class="prose prose-invert">${(ui.narrative||'').replace(/\\n/g,'<br/>')}</div>`;
  document.getElementById('coach-sections').innerHTML = (ui.sections||[]).map(sectionBlock).join('');
}
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

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)

