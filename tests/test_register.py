"""
VQMB — acceptance tests for the Data-Quality Change Register.

One check per register row, each using that row's own acceptance check, plus
the model-review fixes run through the real engine on a synthetic universe.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import pytest

from vqmb import config as C
from vqmb import engine as E
from vqmb import facts as F
from vqmb import fmp as FS
from vqmb import metrics as M
from vqmb import ranking as R
from vqmb import workings as W

from test_acceptance import check, check_blank

NA = float("nan")
T = dt.date.today()


def _d(days_ago):
    return (T - dt.timedelta(days=days_ago)).isoformat()


def _quarters(n, step=91, start=20, **fields):
    """n periodic rows, newest first, every field set to the given values."""
    rows = []
    for k in range(n):
        r = {"date": _d(start + step * k)}
        for f, v in fields.items():
            r[f] = v(k) if callable(v) else v
        rows.append(r)
    return rows


def _blob(**over):
    base = {"profile": [{"sector": "Industrials", "industry": "Machinery", "companyName": "X"}],
            "quote": [{"price": 50.0, "marketCap": 5e10}],
            "income_a": [], "income_q": [], "balance_a": [], "cashflow_a": [],
            "cashflow_q": [], "estimates": [], "ohlcv": []}
    base.update(over)
    return base


def test_register_rows():
    ok = lambda b: "yes" if b else "no"

    # ---- row 1: ROIC for loss-makers (the four cases are in test_roic_fallback);
    #      here, the workings reproduce each case
    for label, f in (("negative return", {"ebit_ttm": -1e9, "invested_capital": 5e10}),
                     ("GP/A stand-in", {"ebit_ttm": 1e9, "invested_capital": -5e9,
                                        "gp_ttm": 2e10, "assets": 1e11}),
                     ("blank", {"ebit_ttm": -1e9, "invested_capital": -5e9})):
        m = M.compute_all(f)
        check(f"row 1: workings reproduce the {label} case",
              ok(W.build(f, m)["q1_roic"]["ok"]), "yes")

    # ---- row 2: V2 blank on negative EV
    check_blank("row 2: EV -5bn, GP 1bn -> V2 blank", M.v2_ev_gp({"ev": -5e9, "gp_ttm": 1e9}))

    # ---- rows 3 + 4: checklist columns exist, and a missing one raises
    df = pd.DataFrame({f"composite_rank_{C.BASE_PROFILE}": [90.0, 20.0], "grp_base": [89.0, 10.0]})
    for _, col, _ in C.CHECKLIST:
        if col and col not in df.columns:
            df[col] = 50.0
    chk = R.checklist(df)
    rnk = [l for l, c, _ in C.CHECKLIST if c == f"composite_rank_{C.BASE_PROFILE}"][0]
    grp = [l for l, c, _ in C.CHECKLIST if c == "grp_base"][0]
    check("row 3: RNK reads the base profile - PASS and FAIL",
          f"{chk[rnk][0]}/{chk[rnk][1]}", "PASS/FAIL")
    check("row 3: GRP reads grp_base - PASS and FAIL", f"{chk[grp][0]}/{chk[grp][1]}", "PASS/FAIL")
    try:
        R.checklist(df.drop(columns=["grp_base"]))
        raised = False
    except KeyError:
        raised = True
    check("row 4: a configured column missing from the frame raises", ok(raised), "yes")

    # ---- row 5: leverage item can fail
    leverage_ok = E.leverage_ok
    lv = leverage_ok(pd.DataFrame({"net_debt": [5e9, -1e9, 5e9], "ebit_ttm": [1e9, 1e9, 1e9],
                                   "is_fin": [False, False, True]}))
    bar = [b for _, c, b in C.CHECKLIST if c == "leverage_ok"][0]
    res = R.checklist(pd.DataFrame({"leverage_ok": lv, **{c: 50.0 for _, c, _ in C.CHECKLIST
                                                            if c and c != "leverage_ok"}}))
    lab = [l for l, c, _ in C.CHECKLIST if c == "leverage_ok"][0]
    check("row 5: levered FAIL, net cash PASS, bank NEUTRAL",
          "/".join(res[lab].tolist()), "FAIL/PASS/NEUTRAL")

    # ---- row 6: one line per issuer
    fold_share_classes = E.fold_share_classes
    out, folded = fold_share_classes(pd.DataFrame(
        {"symbol": ["GOOG", "GOOGL", "MSFT"], "company": ["Alphabet Inc.", "Alphabet Inc.", "Microsoft"],
         "dollar_volume": [1e9, 3e9, 5e9]}))
    check("row 6: a share-class pair leaves one row", len(out), 2)
    check("  ...keeping the more liquid line", ",".join(out["symbol"]), "GOOGL,MSFT")
    check("  ...and names the folded ticker", folded[0]["folded"] if folded else "-", "GOOG")

    # ---- row 8: cadence and contiguity
    half = _quarters(6, step=182, revenue=1e9, operatingIncome=1e8, grossProfit=4e8)
    f = F.build_facts("H", _blob(income_q=half), {}, {})
    check("row 8: half-yearly TTM covers 2 halves (12 months)", f["rev_ttm"], 2e9, 1)
    check("  ...and year-on-year looks 2 rows back", f["yoy_lag"], 2)
    q = _quarters(8, revenue=lambda k: 1e9 + k * 1e7)
    a = F.build_facts("Q", _blob(income_q=q), {}, {})["rev_ttm"]
    b = F.build_facts("Q", _blob(income_q=list(reversed(q))), {}, {})["rev_ttm"]
    check("row 8: rows out of order give the same TTM", ok(a == b and a == a), "yes")
    gap = q[:2] + q[3:]                                   # one quarter missing
    g = F.build_facts("G", _blob(income_q=gap), {}, {})
    check_blank("row 8: a missing quarter blanks TTM", g["rev_ttm"])
    check("  ...and is flagged", ok(g["gapped"]), "yes")

    # ---- row 9: V5 blank on negative EV
    check_blank("row 9: EV -5bn -> V5 blank",
                M.v5_ev_sales_vs_hist({"ev": -5e9, "rev_ttm": 1e10, "ev_sales_median_hist": 2.0}))

    # ---- row 10: one neutral fill in both places
    pcts = pd.DataFrame({k: [80.0] for k, _ in C.QUALITY_ENGINE}
                        | {k: [np.nan] for k, _ in C.QUALITY_SHIELD}
                        | {k: [60.0] for k, _ in C.PRICE_STRENGTH}
                        | {k: [np.nan] for k, _ in C.PRICE_CREDIBILITY})
    pil = R.build_pillars(pd.DataFrame(index=pcts.index), pcts)
    neutral = C.SHIELD_FLOOR + C.SHIELD_SPAN * C.DAMPENER_NEUTRAL_FILL / 100.0
    check("row 10: blank shield -> neutral dampener, not best case",
          float(pil["quality_damp"].iloc[0]), neutral, 1e-9)
    sg = R.safety_grade(pd.Series([np.nan]), pd.Series([100.0]), pd.Series([False]))
    check("  ...and the same neutral fill in the safety grade",
          float(sg["safety_raw"].iloc[0]),
          C.SAFETY_SHIELD_W * C.DAMPENER_NEUTRAL_FILL + C.SAFETY_DRAWDOWN_W * 100.0, 1e-9)

    # ---- row 11: missing EBIT is not a gate; every gate has a cause
    lev, gate = M.q4_leverage_score({"net_debt": 5e9})
    check_blank("row 11: EBIT missing, net debt 5bn -> Q4 blank", lev)
    check("  ...and not gated", ok(gate), "no")
    gdf = pd.DataFrame({"is_fin": [False], "nd_ebit": [np.nan], "leverage_gate": [True],
                        "ebit_negative_with_net_debt": [False]})
    g = R.apply_gates(gdf)
    check("row 11: a Q4 gate always carries a cause",
          ok(bool(g.iloc[0]) and gdf["gate_cause"].iloc[0] != ""), "yes")

    # ---- row 12: EV needs debt and cash
    bal = [{"date": _d(30), "totalStockholdersEquity": 1e10, "totalAssets": 3e10,
            "cashAndShortTermInvestments": 1e9}]           # no debt line
    check_blank("row 12: debt missing -> EV blank", F.build_facts("E", _blob(balance_a=bal), {}, {})["ev"])

    # ---- row 13: PPOP needs all three lines
    check_blank("row 13: a missing line blanks that quarter's PPOP",
                F.volume_line({"revenue": 1e9, "interestExpense": 2e8}, True, "bank"))

    # ---- row 14: missing growth is not zero
    v, tags = M.v3_fwd_earn_yield({"price": 10.0, "rev_ttm": 1e10, "mcap": 1e11,
                                   "net_margin_5y_avg": 0.1, "margin_years": 5})
    check_blank("row 14: no growth figure -> V3 blank", v)
    check("  ...tagged CALC-NA", ok(tags["v3_calc_na"]), "yes")

    # ---- row 15: dilution needs both halves
    check_blank("row 15: stock comp missing -> dilution blank",
                M.hygiene_components({"shares_a0": 1.1e9, "shares_3y": 1e9, "rev_ttm": 1e10})["dilution"])

    # ---- row 16: tangible book needs goodwill
    bal = [{"date": _d(30), "totalStockholdersEquity": 1e10, "totalDebt": 1e9,
            "cashAndShortTermInvestments": 1e9, "totalAssets": 3e10}]
    check_blank("row 16: goodwill missing -> tangible book blank",
                F.build_facts("B", _blob(balance_a=bal), {}, {})["book_tangible"])

    # ---- row 17: P4 ignores missing days
    def bars(closes):
        return [{"date": _d(len(closes) - i), "adjClose": c, "volume": 1e6} for i, c in enumerate(closes)]
    rng = np.random.default_rng(3)
    px = list(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 320)))
    clean = F.build_price_facts(bars(px), {})
    holed = list(px)
    holed[150] = None                                     # one missing day inside the window
    hole = F.build_price_facts(bars(holed), {})
    check("row 17: one missing day does not zero P4",
          ok(hole.get("period_return_sign") in (1.0, -1.0)
             and abs(hole["pct_up_days"] - clean["pct_up_days"]) < 1.0), "yes")

    # ---- row 18: A/D skips missing volume
    from vqmb import darvas as DV
    n = C.AD_SESSIONS + 5
    closes = [100.0 + (i % 2) for i in range(n)]            # alternating up / down days
    vols = [1e6 + i * 1e3 for i in range(n)]
    up_day = next(i for i in range(n - 1, n - C.AD_SESSIONS, -1) if closes[i] > closes[i - 1])
    vols2 = list(vols)
    vols2[up_day] = float("nan")
    up = sum(vols[i] for i in range(n - C.AD_SESSIONS, n) if i != up_day and closes[i] > closes[i - 1])
    dn = sum(vols[i] for i in range(n - C.AD_SESSIONS, n) if closes[i] < closes[i - 1])
    check("row 18: a missing-volume day is excluded, not counted as zero",
          DV.ad_ratio(closes, vols2), up / dn, 1e-12)

    # ---- row 19: a missing count is not enough
    check_blank("row 19: margin_years missing -> V4 blank",
                M.v4_norm_ep({"net_margin_5y_avg": 0.1, "rev_ttm": 1e10, "mcap": 1e11}))
    check("row 19: price_years missing -> SHORT_HISTORY on",
          ok(M.compute_all({"persistence_quarters": 8})["short_history"]), "yes")

    # ---- row 20: the 3-year minimum applies to every 5-year average
    check_blank("row 20: 2 years of ROE -> financial V4 blank",
                M.v4_norm_ep({"is_fin": True, "roe_5y_avg": 0.12, "book": 1e10, "mcap": 2e10, "roe_years": 2}))
    v, _ = M.v3_fwd_earn_yield({"price": 10.0, "rev_ttm": 1e10, "mcap": 1e11, "g_projection": 0.05,
                                "net_margin_5y_avg": 0.1, "margin_years": 2})
    check_blank("row 20: 2 years of margins -> V3 CALC blank", v)

    # ---- row 21: no quarters -> gp_quality "none"
    check("row 21: empty quarters -> gp_quality none",
          F.build_facts("Z", _blob(), {}, {})["gp_quality"], "none")

    # ---- row 22: one basis for a financial margin comparison
    rows = [{"interestIncome": 5e9, "interestExpense": 2e9, "_assets": 1e11,
             "operatingIncome": 1e9, "revenue": 4e9},
            {"operatingIncome": 1e9, "revenue": 4e9, "_assets": 1e11}]   # no interest lines
    check("row 22: a NIM is never compared with an operating margin",
          F.fin_margin_basis(rows, "bank"), "op")

    # ---- row 23: the NTM fallback only uses future fiscal years
    est = [{"date": _d(30), "estimatedEpsAvg": 9.0, "numAnalystsEps": 20},   # ended last month
           {"date": (T + dt.timedelta(days=500)).isoformat(), "estimatedEpsAvg": 3.0,
            "numAnalystsEps": 20}]
    f = F.build_facts("N", _blob(estimates=est), {}, {})
    check("row 23: a fiscal year that ended last month is not used", f["ntm_eps"], 3.0, 1e-9)

    # ---- row 24: currency mismatch sets the value lenses aside
    f = {"fx_mismatch": True, "ebit_ttm": 1e9, "ev": 1e10, "gp_ttm": 2e9}
    m = M.compute_all(f)
    check("row 24: FX mismatch -> V1-V5 blank",
          ok(all(math.isnan(m[k]) for k in ("v1_ebit_ev", "v2_ev_gp", "v4_norm_ep"))), "yes")
    check("  ...and flagged", ok(m["fx_mismatch"]), "yes")

    # ---- row 25: never reuse a partial blob
    tmp = Path(tempfile.mkdtemp())
    calls = {"n": 0}
    orig_get, orig_ep = FS._get, dict(FS.STABLE_ENDPOINTS)
    try:
        FS.STABLE_ENDPOINTS.clear()
        FS.STABLE_ENDPOINTS["profile"] = orig_ep["profile"]
        FS.STABLE_ENDPOINTS["quote"] = orig_ep["quote"]

        def fake_get(url, params, key):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("timeout")
            return [{"symbol": "X", "currency": "USD"}]
        FS._get = fake_get
        FS.fetch_ticker("X", "k", cache_dir=tmp)
        before = calls["n"]
        FS.fetch_ticker("X", "k", cache_dir=tmp)          # cached blob had a failure
        check("row 25: a blob with a failed endpoint is refetched", ok(calls["n"] > before), "yes")
    finally:
        FS._get = orig_get
        FS.STABLE_ENDPOINTS.clear()
        FS.STABLE_ENDPOINTS.update(orig_ep)

    # ---- row 26: an empty market-cap history is never cached
    orig = FS._get
    try:
        FS._get = lambda *a, **k: []
        FS.fetch_extra("EMPTYX", "k", C.HISTORY_YEARS, cache_dir=tmp)
        check("row 26: an empty history leaves no cache file",
              ok(not (tmp / "mcap" / "EMPTYX.json").exists()), "yes")
    finally:
        FS._get = orig

    # ---- row 27: close fallback
    check("row 27: null adjClose falls back to close",
          F._bar_close({"adjClose": None, "close": 42.0}), 42.0, 1e-9)
    check_blank("row 27: a zero close is not a price", F._bar_close({"close": 0}))

    # ---- row 28: income and cash-flow quarters must match
    iq = _quarters(8, revenue=1e9, netIncome=1e8)
    cq = _quarters(8, start=60, operatingCashFlow=1.2e8)     # 40 days out of step
    f = F.build_facts("A", _blob(income_q=iq, cashflow_q=cq), {}, {})
    check_blank("row 28: misaligned cash-flow rows -> operating cash flow TTM blank", f["cfo_ttm"])

    # ---- row 29: period-matched hygiene growth
    f = {"assets": 1.2e10, "assets_3y": 1e10, "rev_a0": 1.1e10, "rev_3y": 1e10, "rev_ttm": 9e10}
    comp = M.hygiene_components(f)
    exp = (1.2 ** (1 / 3) - 1) - (1.1 ** (1 / 3) - 1)
    check("row 29: bloat uses FY0 vs FY-3 on both sides", comp["bloat"], exp, 1e-9)

    # ---- row 30: a composite needs at least half its pillars
    ranks = pd.DataFrame({"v_rank_u": [90.0, 50.0], "q_rank_u": [np.nan, 60.0],
                          "b_rank_u": [np.nan, 40.0], "p_rank_u": [np.nan, 70.0]})
    comps = R.build_composites(ranks)
    check_blank("row 30: one pillar -> no composite", comps[f"composite_raw_{C.BASE_PROFILE}"].iloc[0])
    check("  ...and flagged thin", ok(bool(comps["thin"].iloc[0])), "yes")

    # ---- row 31: an empty benchmark stops the run
    with pytest.raises(E.VQMBError, match="benchmark"):
        E.run_from_data({"AAA": {"blob": _blob()}}, {})

    # ---- row 32: an injected mismatch flags the name
    f = {"ebit_ttm": 1e9, "ev": 1e10}
    m = M.compute_all(f)
    m["v1_ebit_ev"] = 0.5                                  # inject a wrong ranked value
    w = W.build(f, m)
    check("row 32: an injected mismatch is detected", ok(w["v1_ebit_ev"]["ok"] is False), "yes")
    fl = R.compute_flags(pd.DataFrame({"workings_mismatch": [True], "v_rank_u": [1.0], "q_rank_u": [1.0],
                                       "b_rank_u": [1.0], "p_rank_u": [1.0]}))
    check("  ...and flags the name", ok(bool(fl["WKCHK"].iloc[0])), "yes")

    # ---- row 33: statement staleness
    old = _quarters(8, start=380, revenue=1e9)
    check("row 33: year-old statements are flagged stale",
          ok(F.build_facts("S", _blob(income_q=old), {}, {})["stale_statement"]), "yes")

    # ---- row 34: price staleness
    stale = [{"date": _d(30 + 300 - i), "adjClose": 100 + i * 0.1, "volume": 1e6} for i in range(300)]
    pf = F.build_price_facts(stale, {})
    m = M.compute_all({**pf})
    check("row 34: a 30-day-old last bar is stale", ok(pf["price_stale"]), "yes")
    check_blank("  ...and blanks P1-P6", m["p3_high_distance"])

    # ---- row 35: vendor zeros (verified against the cache, switched on)
    z = F.zero_as_missing([{"revenue": 0, "totalAssets": 5}], is_fin=False)[0]
    check("row 35: a vendor 0 revenue reads as missing", ok(z["revenue"] is None), "yes")
    z = F.zero_as_missing([{"revenue": 1e9, "operatingCashFlow": 0}], is_fin=False)[0]
    check("row 35: 0 cash flow on real revenue reads as missing", ok(z["operatingCashFlow"] is None), "yes")
    z = F.zero_as_missing([{"totalDebt": 0, "cashAndShortTermInvestments": 5e9,
                            "goodwillAndIntangibleAssets": 0}], is_fin=False)[0]
    check("row 35: genuine zero debt and zero goodwill are kept",
          ok(z["totalDebt"] == 0 and z["goodwillAndIntangibleAssets"] == 0), "yes")

    # ---- row 36: minimum analysts (verified field, switched on)
    def est(n):
        return [{"date": (T + dt.timedelta(days=100)).isoformat(), "estimatedEpsAvg": 3.0, "numAnalystsEps": n},
                {"date": (T + dt.timedelta(days=465)).isoformat(), "estimatedEpsAvg": 3.3, "numAnalystsEps": n}]
    f = F.build_facts("A", _blob(estimates=est(2)), {}, {})
    check("row 36: a 2-analyst estimate goes to CALC",
          f"{f['fwd_basis']}/{ok(math.isnan(f['ntm_eps']))}", "none/yes")
    f = F.build_facts("A", _blob(estimates=est(12)), {}, {})
    check("row 36: a 12-analyst estimate is used", f["fwd_basis"], "ntm")

    # ---- row 37: implausible daily moves are flagged, not dropped
    px = [100.0] * 100 + [50.0] * 100                         # an unadjusted 2-for-1 split
    pf = F.build_price_facts(bars(px), {})
    check("row 37: a 2-for-1 split bar is flagged for review", len(pf["implausible_moves"]), 1)
    px = [100.0] * 100 + [51.0] * 100                         # the same split on a +2% day: -49%
    check("row 37: a split on an up day (-49%) is still flagged",
          len(F.build_price_facts(bars(px), {})["implausible_moves"]), 1)
    px = [100.0] * 100 + [75.0] * 100                         # a real -25% crash
    check("row 37: an ordinary -25% move is not flagged",
          len(F.build_price_facts(bars(px), {})["implausible_moves"]), 0)

    # ---- row 38: the gross-margin boundary itself
    for gm, want in ((0.994, "kept"), (0.995, "set aside")):
        q = _quarters(8, revenue=1e9, grossProfit=gm * 1e9)
        f = F.build_facts("G", _blob(income_q=q), {}, {})
        check(f"row 38: gross margin {gm:.1%} is {want}",
              "kept" if f["gp_ttm"] == f["gp_ttm"] else "set aside", want)


def _universe(n=40, seed=7):
    """A small synthetic universe run through the real run() and score()."""
    import random
    random.seed(seed)
    rng = np.random.default_rng(seed)
    blobs = {}
    for i in range(n):
        sym = f"S{i:02d}"
        def st(k_n, step, scale, i=i):
            out = []
            for k in range(k_n):
                b = scale * (1 + i * 0.05) * (random.uniform(.9, .98) ** k)
                oi = b * random.uniform(.05, .25)
                out.append({"date": _d(20 + step * k), "revenue": b, "grossProfit": b * random.uniform(.3, .6),
                            "netIncome": oi * .7, "operatingIncome": oi, "weightedAverageShsOutDil": 1e8 * (1 + random.uniform(0, .05) * k),
                            "operatingCashFlow": oi * random.uniform(.5, 1.3), "stockBasedCompensation": b * random.uniform(0, .05),
                            "totalAssets": b * random.uniform(2, 6) * (1.0 + random.uniform(0, .1)) ** -k,
                            "totalDebt": b * random.uniform(0, 2), "totalStockholdersEquity": b * 1.5,
                            "cashAndShortTermInvestments": b * .3, "goodwillAndIntangibleAssets": b * .1,
                            "incomeBeforeTax": oi * .9, "incomeTaxExpense": oi * .2, "interestExpense": b * .01,
                            "operatingExpenses": b * .1})
            return out
        px, p = [], 100.0
        for k in range(1300):
            p *= 1 + rng.normal(0.0004, 0.02)
            px.append({"date": _d(k), "adjClose": p, "close": p, "high": p * 1.01, "low": p * .99, "volume": 1e6})
        blobs[sym] = {"profile": [{"sector": ["Technology", "Industrials"][i % 2], "industry": ["Software", "Machinery"][i % 2],
                                   "companyName": f"Co {sym}"}],
                      "quote": [{"price": px[0]["close"], "marketCap": px[0]["close"] * 1e8}],
                      "income_a": st(6, 365, 4e10), "income_q": st(12, 91, 1e10), "balance_a": st(6, 365, 4e10),
                      "cashflow_q": st(8, 91, 1e10), "estimates": [], "ohlcv": px}
    return blobs


def test_model_review_fixes():
    """Fixes from the model review, run through the real engine."""
    ok = lambda b: "yes" if b else "no"
    blobs = _universe()
    bench = {r["date"]: r["close"] for r in blobs["S00"]["ohlcv"]}

    def load(t):
        return blobs[t], blobs[t]["ohlcv"], {}
    raw, _ = E._build_rows(list(blobs), load, bench, 4)
    df = E.score(raw.copy())

    # 1 - hygiene's sector rank must run the same way as its universe rank
    u, s_ = df["pct_q6_hygiene"], df["pct_q6_hygiene__S"]
    check("review 1: hygiene sector rank runs the same way as the universe rank",
          ok(u.corr(s_) > 0.5), "yes")

    # 2 - hysteresis: yesterday's flags hold a name on between its ON and OFF levels
    prev = pd.DataFrame([{"symbol": s, "HYGIENE": True} for s in df["symbol"]])
    held = E.score(raw.copy(), previous_flags=prev)
    band = (df["q6_hygiene_pct"] >= C.HYGIENE_ON) & (df["q6_hygiene_pct"] < C.HYGIENE_OFF)
    if band.any():
        check("review 2: a name between ON and OFF stays flagged when it was on yesterday",
              ok(bool(held.loc[band, "flag_HYGIENE"].all())), "yes")
    check("review 2: without yesterday's state the same names are not flagged",
          ok(not bool(df.loc[band, "flag_HYGIENE"].any())), "yes")

    # 4 - only endpoints the model reads are fetched
    check("review 4: only the seven endpoints the model reads are fetched",
          ",".join(sorted(FS.STABLE_ENDPOINTS)),
          "balance_a,cashflow_q,estimates,income_a,income_q,profile,quote")


def test_run_from_data_returns_one_frame():
    """The public entry point on the synthetic universe: one DataFrame, every
    stage of the chain present, run health in attrs."""
    blobs = _universe()
    bench = {r["date"]: r["close"] for r in blobs["S00"]["ohlcv"]}
    data = {t: {"blob": b, "ohlcv": b["ohlcv"], "mcap_hist": {}} for t, b in blobs.items()}
    df = E.run_from_data(data, bench, workers=4)
    assert isinstance(df, pd.DataFrame) and len(df) == len(blobs)
    for col in ("v1_ebit_ev", "pct_v1_ebit_ev", "pct_v1_ebit_ev__S", "value_raw", "quality_raw",
                "biz_raw", "price_raw", "v_rank_u", "q_rank_s", f"composite_rank_{C.BASE_PROFILE}",
                "gated", "safety", "flag_GATE", "grp", "checklist_passes", "workings",
                "rev_ttm", "gp_ttm", "net_margin_5y_avg", "vol_yoy_q0"):
        assert col in df.columns, col
    assert df[f"composite_rank_{C.BASE_PROFILE}"].dropna().is_monotonic_decreasing
    for k in ("failed", "excluded_entry_rule", "groups", "market_vitals", "blank_share"):
        assert k in df.attrs, k
    with pytest.raises(ValueError):
        E.run_from_data(data, bench, profile="NOPE")
    again = E.run_from_data(data, bench, previous_flags=df, workers=4)
    assert len(again) == len(df)
