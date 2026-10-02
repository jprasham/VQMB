"""
VQMB — the run.

    import vqmb
    df = vqmb.run(["AAPL", "MSFT", "JPM"], api_key="...")

For every ticker: fetch the vendor data, build the fact sheet, compute the
twenty metrics and their workings, then rank the whole list together:

    metric raws -> percentiles (U and S) -> pillars -> pillar ranks
      -> composites (three profiles) -> gates, conviction, Safety, flags,
         A/D, group strength, checklist

and return one DataFrame, one row per scored name, carrying every input,
intermediate and output of that chain.

Percentiles are relative to the list passed in. Rank the S&P 500 and a name's
ranks are against the S&P 500; rank five names and they are against those five.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import logging
import math
import os
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import config as C
from . import darvas as DV
from . import facts as F
from . import fmp
from . import metrics as M
from . import ranking as R
from . import workings as W
from .groups import assign_group, resolve_merges

log = logging.getLogger("vqmb")


class VQMBError(RuntimeError):
    """The run stopped rather than return numbers that cannot be trusted."""


# fact-sheet keys holding the full daily price series (~1,260 values per name)
_SERIES_KEYS = ("_dates", "_closes", "_highs", "_lows", "_volumes")

# the fact-sheet fields the original row carries, copied in this order
_ROW_FACTS = ("company", "sector", "industry", "price", "mcap", "ev",
              "rev_ttm", "ebit_ttm", "net_debt", "book", "assets",
              "price_years", "ntm_coverage", "ntm_blend", "fwd_basis",
              "dd_episodes", "dd_worst", "persistence_quarters", "gp_quality", "gp_set_aside",
              "dollar_volume", "statement_age_days", "price_age_days", "analyst_count")


# ---------------------------------------------------------------------------
# per-name chain: facts -> metrics -> workings -> unscored overlays
# ---------------------------------------------------------------------------
def _score_name(symbol: str, blob: dict, ohlcv: list, mcap_hist: dict,
                bench: dict) -> tuple[dict, dict]:
    blob = dict(blob)
    blob["ohlcv"] = ohlcv
    f = F.build_facts(symbol, blob, bench, mcap_hist)
    row = M.compute_all(f)
    # every metric's raw inputs, formula and result, checked against the value
    # above - built here while the fact sheet is in hand
    row["_workings"] = W.build(f, row)
    # row 32: a displayed calculation that does not reproduce the ranked value
    # is a defect, not a footnote
    row["workings_mismatch"] = any(e.get("ok") is False for e in row["_workings"].values())
    row["workings_mismatch_keys"] = [k for k, e in row["_workings"].items()
                                     if e.get("ok") is False]
    # carry the identity and diagnostics the chain needs
    for k in [c for c in f if c.startswith("ctx_")]:
        row[k] = f.get(k)
    for k in _ROW_FACTS:
        row[k] = f.get(k)
    # unscored overlays, computed here while the price series is in hand
    cl, hi, lo, vo = (f.get("_closes") or [], f.get("_highs") or [],
                      f.get("_lows") or [], f.get("_volumes") or [])
    row.update(DV.darvas_state(f.get("_dates"), cl, hi, lo, vo))
    row["ad_ratio"] = DV.ad_ratio(cl, vo)
    row["_price_series"] = {"dates": f.get("_dates"), "closes": cl,
                            "highs": hi, "lows": lo, "volumes": vo}
    return row, f


def _build_rows(tickers: list[str], load: Callable[[str], tuple[dict, list, dict]],
                bench: dict, workers: int):
    """Run the per-name chain over every ticker. `load(t)` returns
    (statement blob, ohlcv rows, market-cap history) for one name."""
    done, failed, facts = {"n": 0}, [], {}

    def one(t):
        try:
            blob, ohlcv, mcap_hist = load(t)
            row, f = _score_name(t, blob, ohlcv, mcap_hist, bench)
        except Exception as e:
            failed.append((t, str(e)[:90]))
            return None
        facts[t] = f
        done["n"] += 1
        if done["n"] % 25 == 0 or done["n"] == len(tickers):
            log.info("  %d/%d", done["n"], len(tickers))
        return row

    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        rows = [r for r in ex.map(one, tickers) if r]
    if failed:
        log.warning("%d failed: %s", len(failed), ", ".join(t for t, _ in failed[:8]))
    df = pd.DataFrame(rows)
    df.attrs["failed"] = failed
    return df, facts


# ---------------------------------------------------------------------------
# universe-level steps
# ---------------------------------------------------------------------------
def fold_share_classes(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Register row 6: one line per issuer. Dual share classes (GOOG/GOOGL)
    carry identical statements; ranking both counts the issuer twice in every
    percentile, group median and green count. Keeps the line with the higher
    median dollar volume and returns what was folded."""
    folded = []
    if not C.FOLD_SHARE_CLASSES or "company" not in df.columns:
        return df, folded
    key = df["company"].fillna("").astype(str).str.strip().str.lower()
    dv = pd.to_numeric(df.get("dollar_volume"), errors="coerce").fillna(-1.0)
    keep = pd.Series(True, index=df.index)
    for name, idx in df.groupby(key).groups.items():
        if not name or len(idx) < 2:
            continue
        best = dv.loc[idx].idxmax()
        for i in idx:
            if i != best:
                keep[i] = False
                folded.append({"folded": df.at[i, "symbol"], "kept": df.at[best, "symbol"],
                               "company": df.at[best, "company"]})
    if folded:
        log.info("  folded %d duplicate share class(es): %s", len(folded),
                 ", ".join(f"{x['folded']}->{x['kept']}" for x in folded))
    return df[keep].reset_index(drop=True), folded


def leverage_ok(df: pd.DataFrame) -> pd.Series:
    """Register row 5: a leverage checklist item that can actually fail.
    100 = net cash or ND/EBIT under 1; 0 = ND/EBIT of 1 or more, or a loss with
    net debt; blank when data is missing or the name is a lender, for whom
    ND/EBIT means nothing."""
    nd = pd.to_numeric(df.get("net_debt"), errors="coerce")
    eb = pd.to_numeric(df.get("ebit_ttm"), errors="coerce")
    fin = df["is_fin"].fillna(False).astype(bool)
    out = pd.Series(np.nan, index=df.index)
    out[nd <= 0] = 100.0
    pos = (nd > 0) & (eb > 0)
    out[pos & (nd / eb < 1.0)] = 100.0
    out[pos & (nd / eb >= 1.0)] = 0.0
    out[(nd > 0) & (eb <= 0)] = 0.0
    out[fin] = np.nan
    return out


def score(df: pd.DataFrame, previous_flags: pd.DataFrame | None = None) -> pd.DataFrame:
    """metric raws -> percentiles (U and S) -> pillars -> ranks -> composites
    -> overlays. The chain, in order.

    previous_flags: the previous run's flag state (symbol plus one boolean
    column per flag). Flag hysteresis needs it; without it every flag uses its
    ON threshold.
    """
    df = df.copy()
    df["group_raw"] = [assign_group(s, i)
                       for s, i in zip(df.get("sector", []), df.get("industry", []))]
    # fold sub-scale groups into their nearest neighbour - a median over nine
    # names is not a group read
    merged = resolve_merges(df["group_raw"].value_counts().to_dict(), C.MIN_GROUP_SIZE)
    df["group"] = df["group_raw"].map(merged)
    moved = {g: t for g, t in merged.items() if g != t}
    for g, t in moved.items():
        log.info("  merged group '%s' -> '%s' (under %d members)", g, t, C.MIN_GROUP_SIZE)

    # entry rule: at least a year of price history
    entry = pd.to_numeric(df["price_years"], errors="coerce") >= C.ENTRY_MIN_PRICE_YEARS
    excluded = df[~entry.fillna(False)]
    if len(excluded):
        log.info("  excluded (under 1y of price history): %d", len(excluded))
    df = df[entry.fillna(False)].reset_index(drop=True)

    df, folded = fold_share_classes(df)

    # hygiene: three components ranked first, then averaged into Q6
    pcts = pd.DataFrame(index=df.index)
    for comp in ("accruals", "dilution", "bloat"):
        col = f"hyg_{comp}"
        if col in df.columns:
            # bloat means something different for a lender, and accruals are
            # dropped entirely, so the components splice like the rest
            pcts[f"hyg_{comp}_pct"] = R.percentile_spliced(df, col, -1)
    df["q6_hygiene"] = pcts[[c for c in pcts.columns]].mean(axis=1, skipna=True)
    # hygiene is a mean of percentiles, which only exist now the universe is
    # ranked - so its workings are completed here rather than in build()
    if "_workings" in df.columns:
        for i in df.index:
            wk = df.at[i, "_workings"]
            if isinstance(wk, dict):
                W.finalize_hygiene(wk,
                                   pcts.at[i, "hyg_accruals_pct"] if "hyg_accruals_pct" in pcts else None,
                                   pcts.at[i, "hyg_dilution_pct"] if "hyg_dilution_pct" in pcts else None,
                                   pcts.at[i, "hyg_bloat_pct"] if "hyg_bloat_pct" in pcts else None,
                                   df.at[i, "q6_hygiene"])

    # every ranked metric, universe and sector
    for key, direction in C.ALL_METRICS:
        if key not in df.columns:
            continue
        d = direction
        if key in C.VARIANT_METRICS:
            # different formula for financials -> rank each group against its own
            pcts[key] = R.percentile_spliced(df, key, d,
                                             C.FIN_DIRECTION_OVERRIDE.get(key))
            # U == S by construction for the financial rows
            s_rank = R.percentile_within(df, key, "sector", d)
            fin = df["is_fin"].fillna(False).astype(bool)
            s_rank[fin] = pcts[key][fin]
            pcts[f"{key}__S"] = s_rank
        else:
            pcts[key] = R.percentile(df[key], d)
            pcts[f"{key}__S"] = R.percentile_within(df, key, "sector", d)
    df["q6_hygiene_pct"] = R.percentile(df["q6_hygiene"], +1)
    pcts["q6_hygiene"] = df["q6_hygiene_pct"]
    # The loop above ranked hygiene's SECTOR column in config's lower-is-better
    # direction, but the composite is built from components that are already
    # inverted - higher is cleaner. Rebuild the sector rank the same way as the
    # universe rank, or it reads backwards (the cleanest name ranks 0).
    s_h = R.percentile_within(df, "q6_hygiene", "sector", +1)
    fin_h = df["is_fin"].fillna(False).astype(bool)
    s_h[fin_h] = pcts["q6_hygiene"][fin_h]           # U == S for financials, as elsewhere
    pcts["q6_hygiene__S"] = s_h

    pillars = R.build_pillars(df, pcts)
    df = pd.concat([df, pillars], axis=1)
    ranks = R.rank_pillars(df, sector_col="sector")
    df = pd.concat([df, ranks], axis=1)
    comps = R.build_composites(ranks)
    df = pd.concat([df, comps], axis=1)

    # overlays sit outside every average and are never bought back
    df["gated"] = R.apply_gates(df)
    for profile in C.PROFILES:
        conv = R.conviction(df, profile)
        df[f"green_{profile}"] = conv["green"]
    df["triple"] = R.conviction(df, C.BASE_PROFILE)["triple"]

    dd_pct = R.percentile(df["q5_drawdown"], -1)
    df = pd.concat([df, R.safety_grade(df["quality_shield"], dd_pct, df["gated"])], axis=1)

    prev_flags = None
    if previous_flags is not None:
        prev_flags = previous_flags.set_index("symbol")
        prev_flags = prev_flags.reindex(df["symbol"]).reset_index(drop=True)
    flags = R.compute_flags(df, previous=prev_flags)
    df = pd.concat([df, flags.add_prefix("flag_")], axis=1)
    df["flags"] = R.screen_flags(flags)
    df["leverage_ok"] = leverage_ok(df)

    # group strength for the base profile, computed BEFORE the checklist reads it
    base_groups = R.group_layer(df, C.BASE_PROFILE)
    df["grp_base"] = df["group"].map(base_groups["GRP"])
    df["v_rank_s"] = df["v_rank_s"]
    df["ad"] = DV.ad_grade(df.get("ad_ratio", pd.Series(np.nan, index=df.index)))
    chk = R.checklist(df)
    df["checklist_passes"] = chk["pass_count"]
    for c in chk.columns:
        if c != "pass_count":
            df[f"chk::{c}"] = chk[c]

    df = pd.concat([df, pcts.add_prefix("pct_")], axis=1)
    df.attrs["excluded"] = excluded["symbol"].astype(str).tolist() if len(excluded) else []
    df.attrs["folded"] = folded
    df.attrs["merged_groups"] = moved
    return df


# ---------------------------------------------------------------------------
# market vitals - universe-wide readings, returned in df.attrs
# ---------------------------------------------------------------------------
def _band(value, bands):
    if value is None or not math.isfinite(value):
        return "", "note"
    for upper, verdict, tone in bands:
        if upper is None or value < upper:
            return verdict, tone
    return "", "note"


def market_vitals(df: pd.DataFrame, profile: str) -> dict:
    """The seven universe-wide readings, each with its value and verdict."""
    n = len(df)
    num = lambda c: pd.to_numeric(df.get(c), errors="coerce")
    gated = df.get("gated", pd.Series(False, index=df.index)).fillna(False)
    raw = num(f"composite_raw_{profile}")
    iqr = raw.quantile(0.75) - raw.quantile(0.25) if raw.notna().any() else float("nan")
    st, cr = num("price_strength"), num("price_credibility")
    both = st.notna() & cr.notna()
    corr = st[both].corr(cr[both]) if both.sum() > 10 else float("nan")
    # debt load is an operating-company measure - a bank's "debt" is deposits
    fin = df.get("is_fin", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    nd = num("nd_ebit")[~fin].median()
    greens = int(df.get(f"green_{profile}", pd.Series(False, index=df.index)).fillna(False).sum())
    values = {
        "max5": num("p5_lottery").median() * 100,
        "continuity": num("p4_continuity").median(),
        "nd_ebit": nd,
        "gated": (gated.sum() / n * 100) if n else float("nan"),
        "dispersion": iqr,
        "corr": corr,
        "green": float(greens),
    }
    out = {}
    for key, spec in C.VITALS.items():
        v = values.get(key, float("nan"))
        v = float(v) if v is not None else float("nan")
        verdict, tone = _band(v, spec["bands"])
        if key == "green":
            ceiling = max(1, int(n * C.GREEN_TOP_PCT / 100))
            fill = v / ceiling
            verdict, tone = (("None - no clear leaders", "bad") if v == 0 else
                             ("A few leaders stand out", "note") if fill < 0.5 else
                             ("Leaders stand well clear", "good"))
        out[key] = {"label": spec["label"], "value": v, "unit": spec["unit"],
                    "verdict": verdict, "tone": tone, "what": spec["what"], "why": spec["why"]}
    return out


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------
def _clean_tickers(tickers: Iterable[str]) -> list[str]:
    if isinstance(tickers, str):
        tickers = tickers.split(",")
    out, seen = [], set()
    for t in tickers:
        t = str(t).strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    if not out:
        raise ValueError("no tickers given")
    return out


def _previous_state(previous) -> pd.DataFrame | None:
    """Accept a previous vqmb.run() output (its pipe-separated `flags`
    column), or a table of symbol + one boolean column per flag, and return
    symbol + bare flag names."""
    if previous is None:
        return None
    prev = pd.DataFrame(previous)
    if "symbol" not in prev.columns:
        raise ValueError("previous_flags needs a 'symbol' column")
    if "flags" in prev.columns and not any(
            f"flag_{fl}" in prev.columns or fl in prev.columns for fl in C.FLAG_SEVERITY):
        # a previous vqmb.run() output: `flags` is a pipe-separated list
        on = prev["flags"].fillna("").astype(str).str.split("|")
        state = pd.DataFrame({"symbol": prev["symbol"]})
        for fl in C.FLAG_SEVERITY:
            state[fl] = on.apply(lambda xs, fl=fl: fl in xs)
        return state.drop_duplicates("symbol", keep="last")
    cols = {}
    for fl in C.FLAG_SEVERITY:
        if f"flag_{fl}" in prev.columns:
            cols[f"flag_{fl}"] = fl
        elif fl in prev.columns:
            cols[fl] = fl
    state = prev[["symbol"] + list(cols)].rename(columns=cols)
    return state.drop_duplicates("symbol", keep="last")


# ---------------------------------------------------------------------------
# the output columns (len(COLUMNS))
# ---------------------------------------------------------------------------
# model metric key -> output column name
METRIC_NAMES = {
    "v1_ebit_ev": "ebit_to_ev",
    "v2_ev_gp": "ev_to_gp",
    "v3_fwd_earn_yield": "fwd_earn_yield",
    "v4_norm_ep": "normalized_ep",
    "v5_ev_sales_vs_hist": "ev_sales_vs_hist",
    "q1_roic": "roic",
    "q2_roiic": "roiic",
    "q3_persistence": "growth_persistence",
    "q4_leverage_score": "leverage_score",
    "q5_drawdown": "drawdown_history",
    "q6_hygiene": "hygiene",
    "b1_vs_trend": "latest_q_vs_3y",
    "b2_sequential": "seq_accel",
    "b3_margin_delta": "gm_change_yoy",
    "p1_trend_12_1": "trend_12_1",
    "p2_trend_6_1": "trend_6_1",
    "p3_high_distance": "dist_from_high",
    "p4_continuity": "continuity",
    "p5_lottery": "lottery_days",
    "p6_down_resilience": "down_resilience",
}

# industry group and its strength: the raw taxonomy group, the group after
# sub-scale merges, group strength (GRP) for the active profile, and group
# strength for the base profile (the one the checklist reads)
GROUP_COLUMNS = ["group_raw", "group", "grp", "grp_base"]

# the ten checklist items, named as config.CHECKLIST labels them, then the pass
# count and the leverage item's own 100/0 input
CHECKLIST_COLUMNS = ([f"chk::{label}" for label, _, _ in C.CHECKLIST]
                     + ["checklist_passes", "leverage_ok"])

COLUMNS = (
    ["symbol", "company", "sector", "industry", "country", "price", "mktcap", "fin_mode"]
    + list(METRIC_NAMES.values())
    + [f"u_{n}" for n in METRIC_NAMES.values()]
    + [f"s_{n}" for n in METRIC_NAMES.values()]
    + ["blk_value", "blk_q_engine", "blk_q_shield", "blk_b_mom", "blk_p_strength", "blk_p_cred",
       "shield_dampener", "credibility_dampener", "V_raw", "Q_raw", "B_raw", "P_raw",
       "V", "Q", "B", "P", "s_V", "s_Q", "s_B", "s_P",
       "profile", "composite", "rank"]
    + [f"composite_{p}" for p in C.PROFILES]
    + [f"rank_{p}" for p in C.PROFILES]
    + GROUP_COLUMNS
    + ["gated", "gate_cause", "flags", "pillar_spread", "safety_score", "safety_grade", "ad_grade",
       "green", "triple"]
    + CHECKLIST_COLUMNS
    + [
       "ev", "net_debt", "equity_to_assets", "ntm_coverage", "fwd_basis", "fwd_rev_basis",
       "ntm_blend", "fwd_rev_ntm", "fwd_eps_ntm", "fwd_rev_growth", "fwd_eps_growth", "fwd_pe",
       "ad_ratio", "rev_yoy_ttm", "fwd_earn_yield_calc", "net_debt_to_ebit", "accruals",
       "dilution", "bs_bloat", "rev_cagr_3y", "rev_yoy_q0", "short_history",
       "short_history_metrics", "neutral_fill_metrics"]
)


def _flags(r) -> str:
    """Every active flag, most severe first. GATE is reported in gated/gate_cause."""
    return "|".join(fl for fl in C.FLAG_SEVERITY
                    if fl != "GATE" and bool(r.get(f"flag_{fl}")))


def _short_history_metrics(r) -> str:
    """Which metrics the SHORT_HISTORY test found short of history - the same
    three conditions metrics.compute_all tests."""
    out = []
    if not M._has_at_least(r.get("price_years"), C.DRAWDOWN_MIN_YEARS):
        out.append("drawdown_history")
    if not M._has_at_least(r.get("persistence_quarters"), r.get("persistence_window", 8)):
        out.append("growth_persistence")
    med = r.get("pb_median_hist") if r.get("is_fin") else r.get("ev_sales_median_hist")
    if not M._ok(M._f(med)):
        out.append("ev_sales_vs_hist")
    return "|".join(out)


def to_output(full: pd.DataFrame, profile: str) -> pd.DataFrame:
    """Project the full scored frame onto the output columns in COLUMNS."""
    nan = pd.Series(np.nan, index=full.index)
    num = lambda c: pd.to_numeric(full[c] if c in full.columns else nan, errors="coerce")
    flag = lambda c: (full[c] if c in full.columns else pd.Series(False, index=full.index)) \
        .fillna(False).astype(bool).astype(int)
    rows = [r for _, r in full.iterrows()]
    out = {}                                     # built as one frame at the end
    out["symbol"] = full["symbol"]
    for c in ("company", "sector", "industry", "country"):
        out[c] = full[c] if c in full.columns else None
    out["price"] = num("price")
    out["mktcap"] = num("mcap")
    out["fin_mode"] = flag("is_fin")
    for key, name in METRIC_NAMES.items():
        out[name] = num(key)
    for key, name in METRIC_NAMES.items():
        out[f"u_{name}"] = num(f"pct_{key}")
    for key, name in METRIC_NAMES.items():
        out[f"s_{name}"] = num(f"pct_{key}__S")
    for col, src in (("blk_value", "value_raw"), ("blk_q_engine", "quality_engine"),
                     ("blk_q_shield", "quality_shield"), ("blk_b_mom", "biz_raw"),
                     ("blk_p_strength", "price_strength"), ("blk_p_cred", "price_credibility"),
                     ("shield_dampener", "quality_damp"), ("credibility_dampener", "price_damp"),
                     ("V_raw", "value_raw"), ("Q_raw", "quality_raw"), ("B_raw", "biz_raw"),
                     ("P_raw", "price_raw"), ("V", "v_rank_u"), ("Q", "q_rank_u"),
                     ("B", "b_rank_u"), ("P", "p_rank_u"), ("s_V", "v_rank_s"),
                     ("s_Q", "q_rank_s"), ("s_B", "b_rank_s"), ("s_P", "p_rank_s")):
        out[col] = num(src)
    out["profile"] = profile
    out["composite"] = num(f"composite_raw_{profile}")
    out["rank"] = num(f"composite_rank_{profile}")
    for p in C.PROFILES:
        out[f"composite_{p}"] = num(f"composite_raw_{p}")
    for p in C.PROFILES:
        out[f"rank_{p}"] = num(f"composite_rank_{p}")
    for c in ("group_raw", "group"):
        out[c] = full[c] if c in full.columns else None
    out["grp"] = num("grp")
    out["grp_base"] = num("grp_base")
    out["gated"] = flag("gated")
    # the cause ranking.apply_gates recorded - the single source for gates
    out["gate_cause"] = (full["gate_cause"] if "gate_cause" in full.columns
                         else pd.Series("", index=full.index)).fillna("").astype(str)
    out["flags"] = [_flags(r) for r in rows]
    out["pillar_spread"] = num("flag_pillar_spread")
    out["safety_score"] = num("safety_raw")
    out["safety_grade"] = full["safety"]
    out["ad_grade"] = full["ad"]
    out["green"] = flag(f"green_{profile}")
    out["triple"] = flag("triple")
    for label, _, _ in C.CHECKLIST:
        out[f"chk::{label}"] = full[f"chk::{label}"] if f"chk::{label}" in full.columns else None
    out["checklist_passes"] = num("checklist_passes")
    out["leverage_ok"] = num("leverage_ok")
    out["ev"] = num("ev")
    out["net_debt"] = num("net_debt")
    out["equity_to_assets"] = num("capital_ratio")
    out["ntm_coverage"] = num("ntm_coverage")
    out["fwd_basis"] = full["fwd_basis"] if "fwd_basis" in full.columns else None
    out["fwd_rev_basis"] = full["fwd_rev_basis"] if "fwd_rev_basis" in full.columns else None
    out["ntm_blend"] = full["ntm_blend"] if "ntm_blend" in full.columns else None
    out["fwd_rev_ntm"] = num("fwd_rev")
    out["fwd_eps_ntm"] = num("ntm_eps")
    out["fwd_rev_growth"] = num("ctx_fwd_rev_growth")
    out["fwd_eps_growth"] = num("ctx_fwd_eps_growth")
    eps = out["fwd_eps_ntm"]
    out["fwd_pe"] = (out["price"] / eps).where(eps > 0)       # meaningless on a forecast loss
    out["ad_ratio"] = num("ad_ratio")
    out["rev_yoy_ttm"] = num("rev_yoy_ttm")
    loss, calc = flag("v3_loss"), flag("v3_calc")
    out["fwd_earn_yield_calc"] = np.where(loss == 1, 2, np.where(calc == 1, 1, 0))
    out["net_debt_to_ebit"] = num("nd_ebit")
    out["accruals"] = num("hyg_accruals")
    out["dilution"] = num("hyg_dilution")
    out["bs_bloat"] = num("hyg_bloat")
    # the two legs of latest_q_vs_3y, on the volume line the model uses
    out["rev_cagr_3y"] = num("vol_cagr_3y")
    out["rev_yoy_q0"] = num("vol_yoy_q0")
    out["short_history"] = flag("short_history")
    out["short_history_metrics"] = [_short_history_metrics(r) for r in rows]
    # which ranked metrics had no percentile and so counted as the neutral 50
    if C.METRIC_NEUTRAL_FILL_ON:
        blank = {name: num(f"pct_{key}").isna() for key, name in METRIC_NAMES.items()}
        out["neutral_fill_metrics"] = ["|".join(n for n, b in blank.items() if b.iloc[i])
                                       for i in range(len(full))]
    else:
        out["neutral_fill_metrics"] = ""
    assert list(out) == COLUMNS
    out = pd.DataFrame(out, index=full.index)
    out = out.sort_values("rank", ascending=False, na_position="last").reset_index(drop=True)
    out.attrs = dict(full.attrs)
    return out


def _run_full(data: dict, benchmark: dict, profile: str = C.BASE_PROFILE,
              previous_flags=None, workers: int = 8) -> pd.DataFrame:
    """The whole chain with every intermediate column kept (about 300). The
    public functions project this onto the output columns in COLUMNS."""
    if profile not in C.PROFILES:
        raise ValueError(f"profile must be one of {', '.join(C.PROFILES)}")
    if not benchmark:
        # row 31: an empty benchmark would blank P6 for the whole universe
        # without a trace - stop instead
        raise VQMBError("the benchmark price series came back empty")
    F.TODAY = dt.date.today()
    tickers = _clean_tickers(data.keys())
    data = {str(k).strip().upper(): v for k, v in data.items()}

    def load(t):
        d = data[t]
        if d.get("error"):
            raise RuntimeError(d["error"])          # the fetch itself failed
        return d.get("blob") or {}, d.get("ohlcv") or [], d.get("mcap_hist") or {}

    raw, facts = _build_rows(tickers, load, benchmark, workers)
    failed = raw.attrs.get("failed", [])
    if raw.empty:
        raise VQMBError("no ticker could be scored: "
                        + "; ".join(f"{t}: {e}" for t, e in failed[:10]))
    df = score(raw, previous_flags=_previous_state(previous_flags))
    excluded, folded, moved = (df.attrs.get("excluded", []), df.attrs.get("folded", []),
                               df.attrs.get("merged_groups", {}))

    # row 32: a displayed calculation that does not reproduce the ranked value
    # is a defect. A few are flagged on the name (WKCHK); more than the
    # configured share means something systematic broke, so the run stops.
    mism = df.get("workings_mismatch", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    if len(df) and mism.mean() > C.WORKINGS_MISMATCH_ABORT_SHARE:
        bad = df.loc[mism, "symbol"].astype(str).tolist()
        raise VQMBError(f"{mism.sum()} of {len(df)} names ({mism.mean():.1%}) have a workings "
                        f"mismatch, above the {C.WORKINGS_MISMATCH_ABORT_SHARE:.0%} limit: {bad[:15]}")

    df = df.drop(columns=["_price_series"])
    groups = R.group_layer(df, profile)
    # group strength for the active profile, rounded as FLUX publishes it
    df = pd.concat([df, df["group"].map(groups["GRP"]).round(0).rename("grp")], axis=1)

    # every fact-sheet field the rows do not already carry
    extra = {}
    for i, sym in df["symbol"].items():
        f = facts.get(sym) or {}
        for k, v in f.items():
            if k in _SERIES_KEYS or k in df.columns:
                continue
            extra.setdefault(k, {})[i] = v
    if extra:
        df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)

    df.attrs = {
        "as_of": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "profile": profile,
        "names_in": len(tickers), "names_scored": int(len(df)),
        "gated": int(df["gated"].fillna(False).sum()),
        "green": int(df[f"green_{profile}"].fillna(False).sum()),
        "triple": int(df["triple"].fillna(False).sum()),
        "fin_variant": int(df["is_fin"].fillna(False).sum()),
        "failed": failed,
        "excluded_entry_rule": excluded,
        "folded_share_classes": folded,
        "merged_groups": moved,
        "blank_share": {METRIC_NAMES[k]: round(float(pd.to_numeric(df[k], errors="coerce").isna().mean()), 4)
                        for k, _ in C.ALL_METRICS if k in df.columns},
        "data_flags": {fl: int(df.get(f"flag_{fl}", pd.Series(False)).fillna(False).astype(bool).sum())
                       for fl in ("NOGP", "GPA", "FX", "STALE", "THIN", "PXCHK", "WKCHK", "CALC")},
        "cadence": {str(k): int(v) for k, v in df.get("cadence", pd.Series(dtype=object))
                    .fillna("unknown").value_counts().items()},
        "workings_mismatch": df.loc[df["workings_mismatch"].fillna(False).astype(bool),
                                    "symbol"].astype(str).tolist(),
        "implausible_moves": {str(r["symbol"]): r["implausible_moves"]
                              for _, r in df.iterrows()
                              if isinstance(r.get("implausible_moves"), list) and r["implausible_moves"]},
        "groups": groups.reset_index().to_dict(orient="records"),
        "market_vitals": market_vitals(df, profile),
    }
    return df


def run_from_data(data: dict, benchmark: dict, profile: str = C.BASE_PROFILE,
                  previous_flags=None, workers: int = 8) -> pd.DataFrame:
    """Score data already in hand - no API calls.

    data: {ticker: {"blob": statement blob as fmp.fetch_ticker returns it,
                    "ohlcv": daily bars, "mcap_hist": {date: market cap}}}
    benchmark: {date: close} for SPY.
    Other arguments as run(). Returns the DataFrame of COLUMNS.
    """
    full = _run_full(data, benchmark, profile=profile, previous_flags=previous_flags,
                     workers=workers)
    return to_output(full, profile)


def fetch_data(tickers: Iterable[str], api_key: str | None = None, workers: int = 8,
               cache_dir: str | os.PathLike | None = None, refresh: bool = False) -> dict:
    """Fetch every name's vendor data from FMP. Returns the `data` mapping
    run_from_data() takes. Useful when the same fetch is to be scored more
    than once, or saved."""
    api_key = _key(api_key)
    tickers = _clean_tickers(tickers)

    def one(t):
        try:
            blob = fmp.fetch_ticker(t, api_key, cache_dir=cache_dir, refresh=refresh)
            extra = fmp.fetch_extra(t, api_key, C.HISTORY_YEARS, cache_dir=cache_dir)
            return t, {"blob": blob, "ohlcv": extra["ohlcv"], "mcap_hist": extra["mcap_hist"]}
        except Exception as e:
            log.warning("%s: %s", t, e)
            return t, {"blob": {}, "ohlcv": [], "mcap_hist": {}, "error": str(e)}

    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return dict(ex.map(one, tickers))


def fetch_benchmark(api_key: str | None = None) -> dict:
    """SPY daily closes over the model's history window, as {date: close}."""
    return fmp.benchmark_closes(_key(api_key), C.HISTORY_YEARS)


def index_constituents(index: str, api_key: str | None = None) -> list[str]:
    """Tickers of 'sp500', 'nasdaq', 'dowjones' or 'etf:<TICKER>' from FMP."""
    return fmp.fetch_index_constituents(index, _key(api_key))


def _key(api_key: str | None) -> str:
    key = api_key or os.environ.get("FMP_API_KEY", "")
    if not key:
        raise ValueError("an FMP API key is required: pass api_key=... "
                         "or set the FMP_API_KEY environment variable")
    return key


def run(tickers: Iterable[str], api_key: str | None = None, *,
        profile: str = C.BASE_PROFILE, previous_flags=None, workers: int = 8,
        cache_dir: str | os.PathLike | None = None, refresh: bool = False) -> pd.DataFrame:
    """Score a list of tickers with the VQMB model.

    Parameters
    ----------
    tickers : list of str (or one comma-separated string)
        The universe. Every percentile and rank is relative to this list.
    api_key : str, optional
        Your FMP API key. Falls back to the FMP_API_KEY environment variable.
    profile : "TRADER" | "PM" | "GROWTH"
        The active profile: fills `composite`, `rank` and `green`, and sorts
        the output. All three composites are always computed.
    previous_flags : DataFrame, optional
        A previous run's output (or any table with `symbol` and the flag
        columns). Gives HYGIENE / DISCRETE / CONFLICT their hysteresis.
    workers : int
        Parallel fetch/compute threads.
    cache_dir : path, optional
        Cache raw FMP responses here (fundamentals 1 day, market-cap history
        7 days; prices are always fresh). Default: no caching.
    refresh : bool
        Ignore cached fundamentals and refetch.

    Returns
    -------
    pandas.DataFrame
        The columns in COLUMNS, one row per scored name, sorted by `rank`.
        Run-level information (failures, exclusions, group table, market
        vitals, blank shares) is in `df.attrs`.
    """
    api_key = _key(api_key)
    tickers = _clean_tickers(tickers)
    log.info("VQMB — %d names, %d metrics", len(tickers), C.METRIC_COUNT)
    bench = fmp.benchmark_closes(api_key, C.HISTORY_YEARS)
    if not bench:
        raise VQMBError("the benchmark price series came back empty")
    data = fetch_data(tickers, api_key, workers=workers, cache_dir=cache_dir, refresh=refresh)
    return run_from_data(data, bench, profile=profile, previous_flags=previous_flags,
                         workers=workers)
