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
    check_blank("row 2: EV -5bn, GP 1bn -> V2 blank", M.v2_gp_ev({"ev": -5e9, "gp_ttm": 1e9}))

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
    pcts = pd.DataFrame({k: [80.0] for k, _ in C.QUALITY_METRICS}
                        | {k: [60.0] for k, _ in C.PRICE_STRENGTH}
                        | {k: [np.nan] for k, _ in C.PRICE_CREDIBILITY})
    pil = R.build_pillars(pd.DataFrame(index=pcts.index), pcts)
    neutral = C.CRED_FLOOR + C.CRED_SPAN * C.DAMPENER_NEUTRAL_FILL / 100.0
    check("row 10: blank credibility -> neutral dampener, not best case",
          float(pil["price_damp"].iloc[0]), neutral, 1e-9)
    sg = R.safety_grade(pd.Series([np.nan]), pd.Series([100.0]), pd.Series([False]))
    check("  ...and the same neutral fill in the safety grade",
          float(sg["safety_raw"].iloc[0]),
          C.SAFETY_PROTECTION_W * C.DAMPENER_NEUTRAL_FILL + C.SAFETY_DRAWDOWN_W * 100.0, 1e-9)

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
          ok(all(math.isnan(m[k]) for k in ("v1_ebit_ev", "v2_gp_ev", "v4_norm_ep"))), "yes")
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


def test_output_is_the_documented_columns():
    """The public entry point returns exactly the 146 documented columns, and
    each one carries the model's own value."""
    blobs = _universe()
    bench = {r["date"]: r["close"] for r in blobs["S00"]["ohlcv"]}
    data = {t: {"blob": b, "ohlcv": b["ohlcv"], "mcap_hist": {}} for t, b in blobs.items()}
    df = E.run_from_data(data, bench, workers=4)
    assert list(df.columns) == E.COLUMNS and len(E.COLUMNS) == 146
    assert len(df) == len(blobs)
    assert df["rank"].dropna().is_monotonic_decreasing
    for k in ("failed", "excluded_entry_rule", "groups", "market_vitals", "blank_share"):
        assert k in df.attrs, k

    full = E._run_full(data, bench, workers=4).set_index("symbol")
    out = df.set_index("symbol").loc[full.index]
    same = lambda a, b: bool(((a == b) | (a.isna() & b.isna())).all())
    for key, name in E.METRIC_NAMES.items():
        assert same(out[name], full[key]), name
        assert same(out[f"u_{name}"], full[f"pct_{key}"]), name
        assert same(out[f"s_{name}"], full[f"pct_{key}__S"]), name
    for col, src in (("V", "v_rank_u"), ("Q_raw", "quality_raw"), ("P_raw", "price_raw"),
                     ("composite", f"composite_raw_{C.BASE_PROFILE}"),
                     ("rank", f"composite_rank_{C.BASE_PROFILE}"), ("safety_score", "safety_raw"),
                     ("net_debt_to_ebit", "nd_ebit"), ("accruals", "hyg_accruals"),
                     ("s_V", "v_rank_s"), ("s_Q", "q_rank_s"), ("s_B", "b_rank_s"),
                     ("s_P", "p_rank_s"), ("grp_base", "grp_base"), ("grp", "grp"),
                     ("checklist_passes", "checklist_passes"), ("leverage_ok", "leverage_ok")):
        assert same(out[col], full[src]), col
    assert (out["safety_grade"] == full["safety"]).all()
    # every gated name names its gate, and no ungated name does - and the
    # cause is the one ranking.apply_gates recorded, not a re-derivation
    assert ((out["gated"] == 1) == (out["gate_cause"] != "")).all()
    assert (out["gate_cause"] == full["gate_cause"].fillna("")).all()
    # every name carries its group, and grp is that group's strength
    assert (out["group"] == full["group"]).all() and (out["group_raw"] == full["group_raw"]).all()
    grp = {g["group"]: g["GRP"] for g in df.attrs["groups"]}
    assert same(out["grp"], out["group"].map(grp).round(0)), "grp"
    # the ten checklist items, as config names them, and the pass count adds up
    chk = [f"chk::{label}" for label, _, _ in C.CHECKLIST]
    for c in chk:
        assert (out[c] == full[c]).all(), c
        assert out[c].isin(["PASS", "NEUTRAL", "FAIL"]).all(), c
    assert ((out[chk] == "PASS").sum(axis=1) == out["checklist_passes"]).all()
    # neutral_fill_metrics names exactly the metrics with a blank u_ percentile
    # in a pillar that has at least one real metric (and that apply to the name)
    for keys in R.PILLAR_METRICS.values():
        names = [E.METRIC_NAMES[k] for k in keys]
        has_real = out[[f"u_{n}" for n in names]].notna().any(axis=1)
        for k, name in zip(keys, names):
            na = (out["fin_mode"] == 1) & (k in C.FIN_NOT_APPLICABLE)
            listed = out["neutral_fill_metrics"].str.split("|").apply(lambda xs, n=name: n in xs)
            assert (listed == (out[f"u_{name}"].isna() & has_real & ~na)).all(), name
    # short_history_metrics is filled exactly when short_history is on
    assert ((out["short_history"] == 1) == (out["short_history_metrics"] != "")).all()

    with pytest.raises(ValueError):
        E.run_from_data(data, bench, profile="NOPE")
    # yesterday's output feeds hysteresis directly
    prev = df.copy()
    prev["flags"] = "HYGIENE"
    held = E.run_from_data(data, bench, previous_flags=prev, workers=4).set_index("symbol")
    band = (full["q6_hygiene_pct"] >= C.HYGIENE_ON) & (full["q6_hygiene_pct"] < C.HYGIENE_OFF)
    assert held.loc[band[band].index, "flags"].str.contains("HYGIENE").all()


def test_analyst_count_not_fx_converted(monkeypatch):
    """Analyst counts on estimate rows are head counts: a cross-currency name
    must keep them whole while the money fields are converted."""
    monkeypatch.setattr(FS, "fx_rate", lambda base, quote, key: 1.124)
    monkeypatch.setattr(FS, "fx_series", lambda base, quote, key: {})
    blob = {
        "profile": [{"currency": "USD"}],
        "income_a": [{"date": _d(100), "reportedCurrency": "EUR", "revenue": 100.0}],
        "estimates": [{"date": _d(-200), "epsAvg": 10.0, "estimatedEpsAvg": 10.0,
                       "revenueAvg": 50.0, "numAnalystsEps": 11, "numAnalystsRevenue": 25}],
    }
    out = FS.normalise_currency(blob, "ASML", "k")
    est = out["estimates"][0]
    assert est["numAnalystsEps"] == 11 and est["numAnalystsRevenue"] == 25
    assert abs(est["estimatedEpsAvg"] - 11.24) < 1e-9 and abs(est["revenueAvg"] - 56.2) < 1e-9
    assert out["income_a"][0]["revenue"] == pytest.approx(112.4)


def test_blank_sector_or_industry_does_not_stop_the_run():
    """A blank industry or sector reaches assign_group as NaN (a float). It must
    land in a group (or UNMAPPED), never raise and stop every other name."""
    from vqmb.groups import assign_group, UNMAPPED
    assert assign_group("Technology", NA) == assign_group("Technology", None)
    assert assign_group(NA, NA) == UNMAPPED
    assert assign_group(NA, "No Such Industry") == UNMAPPED
    blobs = _universe()
    bench = {r["date"]: r["close"] for r in blobs["S00"]["ohlcv"]}
    data = {t: {"blob": json.loads(json.dumps(b)), "ohlcv": b["ohlcv"], "mcap_hist": {}}
            for t, b in blobs.items()}
    prof = data["S07"]["blob"]["profile"]
    prof = prof[0] if isinstance(prof, list) else prof
    prof["industry"], prof["sector"] = None, None
    out = E.run_from_data(data, bench, workers=1)
    assert len(out) == len(blobs)
    assert out.set_index("symbol").at["S07", "group_raw"] == UNMAPPED


# --------------------------------------------------------------------------
# probe-confirmed fixes (Oct 2026 FMP probe)
# --------------------------------------------------------------------------
def _data():
    blobs = _universe()
    bench = {r["date"]: r["close"] for r in blobs["S00"]["ohlcv"]}
    data = {t: {"blob": json.loads(json.dumps(b)), "ohlcv": json.loads(json.dumps(b["ohlcv"])),
                "mcap_hist": {}} for t, b in blobs.items()}
    return data, bench


def _with_estimates(blob, eps=(6.0, 7.0), n=10):
    blob["estimates"] = [{"date": _d(-120), "estimatedEpsAvg": eps[0], "numAnalystsEps": n},
                         {"date": _d(-485), "estimatedEpsAvg": eps[1], "numAnalystsEps": n}]


def _scale_bars(rows, k):
    for r in rows:
        for f in ("open", "high", "low", "close", "adjClose"):
            if r.get(f) is not None:
                r[f] = r[f] * k


def test_minor_unit_price_is_scaled_to_the_major_unit(monkeypatch):
    """Probe: AZN.L is quoted in GBp (11870) with a GBP market cap, so EPS / price
    came out 100x too small. A pence line must score exactly like the same
    company quoted in pounds."""
    monkeypatch.setattr(FS, "fx_rate", lambda base, quote, key: 1.0 if base == quote else 1.3)
    monkeypatch.setattr(FS, "fx_series", lambda base, quote, key: {})
    data, bench = _data()
    _with_estimates(data["S07"]["blob"])
    base = E.run_from_data(data, bench, workers=1).set_index("symbol").loc["S07"]

    data, bench = _data()
    blob = data["S07"]["blob"]
    _with_estimates(blob)
    blob["profile"][0]["currency"] = "GBp"
    for r in blob["income_a"] + blob["income_q"]:
        r["reportedCurrency"] = "GBP"
    blob["quote"][0]["price"] *= 100                 # pence; market cap stays in pounds
    _scale_bars(data["S07"]["ohlcv"], 100)
    blob = FS.normalise_currency(blob, "S07", "k")
    assert blob["_fx"]["quote_currency"] == "GBP" and blob["_price_unit_scale"] == 0.01
    data["S07"]["blob"] = blob
    out = E.run_from_data(data, bench, workers=1).set_index("symbol").loc["S07"]
    for c in ("price", "fwd_earn_yield", "fwd_pe", "ebit_to_ev", "dist_from_high", "trend_12_1", "rank"):
        assert out[c] == pytest.approx(base[c], rel=1e-9, nan_ok=True), c
    assert "PXCHK" not in out["flags"]


def test_minor_unit_statements_convert_to_the_major_currency(monkeypatch):
    """Probe: /quote USDZAC exists (rand cents), so ZAc statements were
    converted into cents against a market cap in rand."""
    seen = []
    monkeypatch.setattr(FS, "fx_rate", lambda b, q, k: seen.append((b, q)) or 18.0)
    monkeypatch.setattr(FS, "fx_series", lambda b, q, k: {})
    blob = {"profile": [{"currency": "ZAc"}],
            "income_a": [{"date": _d(100), "reportedCurrency": "USD", "revenue": 10.0}]}
    out = FS.normalise_currency(blob, "NPN.JO", "k")
    assert ("USD", "ZAR") in seen and ("USD", "ZAC") not in seen
    assert out["income_a"][0]["revenue"] == pytest.approx(180.0)
    assert out["_price_unit_scale"] == 0.01


def test_duplicate_estimate_dates_do_not_fail_the_name():
    """Probe: 6758.T failed with "'<' not supported between instances of
    'dict' and 'dict'" - two estimate rows on one date."""
    data, bench = _data()
    blob = data["S07"]["blob"]
    _with_estimates(blob)
    dup = dict(blob["estimates"][0], estimatedEpsAvg=6.5)  # same date, different row
    blob["estimates"] = blob["estimates"] + [dup]
    out = E.run_from_data(data, bench, workers=1)
    assert out.attrs["failed"] == []
    assert out.set_index("symbol").at["S07", "fwd_basis"] == "ntm"


def test_share_basis_restates_eps_per_traded_unit():
    """Probe: HDB market cap / price is 2.99x the reported share count, so EPS
    per reported share was being divided by a price per ADR."""
    data, bench = _data()
    _with_estimates(data["S07"]["blob"])
    base = E.run_from_data(data, bench, workers=1).set_index("symbol").loc["S07"]

    data, bench = _data()
    blob = data["S07"]["blob"]
    _with_estimates(blob)
    blob["quote"][0]["price"] *= 3                   # one traded unit = three reported shares
    _scale_bars(data["S07"]["ohlcv"], 3)
    full = E._run_full(data, bench, workers=1).set_index("symbol").loc["S07"]
    out = E.to_output(E._run_full(data, bench, workers=1), C.BASE_PROFILE).set_index("symbol").loc["S07"]
    assert full["share_basis_restated"] and full["share_basis"] == pytest.approx(3.0, rel=1e-6)
    assert out["fwd_earn_yield"] == pytest.approx(base["fwd_earn_yield"], rel=1e-9)
    assert out["fwd_eps_ntm"] == pytest.approx(3 * base["fwd_eps_ntm"], rel=1e-9)
    assert not full["workings_mismatch"]


def test_fold_compares_turnover_in_one_currency():
    """Probe: TSM/2330.TW folded to 2330.TW because TWD turnover was compared
    with USD turnover. The comparison is in config.TURNOVER_CURRENCY."""
    data, bench = _data()
    data["S07"]["blob"]["profile"][0]["currency"] = "USD"
    twin = json.loads(json.dumps(data["S07"]))
    twin["blob"]["profile"][0]["currency"] = "TWD"
    twin["blob"]["_turnover_rate"] = 0.0314           # USD per TWD
    for r in twin["ohlcv"]:
        r["volume"] *= 10                             # 10x the local turnover, 0.3x in USD
    data["S07L"] = twin
    out = E.run_from_data(data, bench, workers=1)
    kept = [x["kept"] for x in out.attrs["folded_share_classes"]]
    assert kept == ["S07"]


def test_client_errors_are_not_retried(monkeypatch):
    """Probe: every /forex-quote call was a 404; the library slept through
    four retries on each before moving on."""
    calls, sleeps = [], []

    class R:
        status_code = 404
        text = "not found"

    monkeypatch.setattr(FS.requests, "get", lambda *a, **k: calls.append(1) or R())
    monkeypatch.setattr(FS.time, "sleep", lambda s: sleeps.append(s))
    with pytest.raises(FS.PermanentHTTPError):
        FS._get(FS.STABLE_BASE + "/quote", {"symbol": "X"}, "k")
    assert len(calls) == 1 and sleeps == []


def test_fx_cache_is_reset_per_fetch(monkeypatch):
    FS._FX_CACHE[("ILS", "USD")] = None              # a miss from an earlier run
    monkeypatch.setattr(FS, "fetch_ticker", lambda *a, **k: {})
    monkeypatch.setattr(FS, "fetch_extra", lambda *a, **k: {"ohlcv": [], "mcap_hist": {}})
    E.fetch_data(["X"], api_key="k", workers=1)
    assert ("ILS", "USD") not in FS._FX_CACHE


# --------------------------------------------------------------------------
# Model changes, October 2026
# --------------------------------------------------------------------------
def _est(days_ahead, rev, n_rev=10):
    d = (F.TODAY + dt.timedelta(days=days_ahead))
    return {"date": d.isoformat(), "estimatedRevenueAvg": rev, "numAnalystsRevenue": n_rev}


def test_v2_is_gross_profit_over_ev_and_ranks_unchanged():
    check("V2 = GP / EV", M.v2_gp_ev({"gp_ttm": 2e9, "ev": 8e9}), 0.25, 1e-12)
    check_blank("GP <= 0 -> blank", M.v2_gp_ev({"gp_ttm": -1e8, "ev": 8e9}))
    check_blank("EV <= 0 -> blank", M.v2_gp_ev({"gp_ttm": 1e9, "ev": -8e9}))
    check("financials keep book / price", M.v2_gp_ev({"is_fin": True, "book": 5e9, "mcap": 1e10}), 0.5)
    gp = pd.Series([1.0, 2.0, 3.0, 5.0])
    ev = pd.Series([10.0, 9.0, 30.0, 12.0])
    old = R.percentile(ev / gp, -1)
    new = R.percentile(gp / ev, dict(C.VALUE_METRICS)["v2_gp_ev"])
    assert np.allclose(old, new), "every V2 percentile is unchanged"
    assert C.FIN_DIRECTION_OVERRIDE == {}


# V4 archetypes: (revenue, net income, TTM revenue, market cap, extra facts)
_NE = {
    "micron_peak": ([21.44, 27.71, 30.76, 15.54, 25.11, 37.38, 133.19],
                    [2.69, 5.86, 8.69, -5.83, 0.778, 8.54, 84.97], 282.31, 1160, {}),
    "micron_3y": ([25.11, 37.38, 133.19], [0.778, 8.54, 84.97], 282.31, 1160, {}),
    "micron_fy23": ([21.44, 27.71, 30.76, 15.54], [2.69, 5.86, 8.69, -5.83], 25.11, 75, {}),
    "acme": ([100, 80, 120, 140, 160], [8, 2, 12, 17, 22], 200, 200, {}),
    "stable": ([143, 168, 198, 212, 245, 281, 318], [44, 61, 73, 72, 88, 101, 115], 360, 3500, {}),
    "hypergrowth": ([100, 170, 260, 360, 460, 560, 660], [-15, -10, 0, 25, 60, 95, 130], 780, 4000, {}),
    "decliner": ([78, 79, 63, 54, 53, 53], [21, 20, 8, 1.7, -18.8, -0.3], 55, 150, {}),
    "acquirer": ([100, 108, 116, 125, 135, 146, 300], [30, 32, 35, 37, 40, 44, 90], 320, 1500, {}),
    "quiet_semi": ([100, 105, 111, 117, 124, 131, 139], [20, 21, 22, 24, 25, 27, 29], 150, 600,
                   {"sector": "Technology", "industry": "Semiconductors"}),
    "impairment": ([143, 168, 198, 212, 245, 281, 318], [44, 61, 73, 72, 20, 101, 115], 360, 3500,
                   {"ne_oi": [57, 79, 95, 94, 114, 131, 150]}),
    "short": ([100, 110], [10, 11], 120, 100, {}),
}


def _ne_facts(k):
    r, n, c, mc, kw = _NE[k]
    return {"ne_rev": r, "ne_ni": n, "ne_oi": [NA] * len(r), "rev_ttm": c, "mcap": mc,
            "ne_dates": [f"20{19 + i}-12-31" for i in range(len(r))], **kw}


def test_v4_normalized_earnings_is_one_sided():
    res = {k: M.normalized_earnings(_ne_facts(k)) for k in _NE}

    def naive(k):
        r, n, c, mc, _ = _NE[k]
        return c * max(min(n[-1] / r[-1], .5), -.5) / mc

    above = [k for k, x in res.items() if not math.isnan(x["yield"]) and x["yield"] > naive(k) + 1e-12]
    assert above == [], "one-sided: never above today's run rate"
    assert res["micron_peak"]["yield"] < 0.25 * naive("micron_peak"), "a peak is cut by > 75%"
    assert res["micron_3y"]["cyclical"] and "margin_jump" in res["micron_3y"]["why"]
    assert res["acme"]["cyclical"] and "rev_fall" in res["acme"]["why"]
    for k in ("stable", "hypergrowth", "acquirer"):
        assert not res[k]["cyclical"], k
        check(f"{k}: not cyclical, equals today's run rate", res[k]["yield"], naive(k), 1e-9)
    assert res["quiet_semi"]["why"] == "industry"
    assert not res["impairment"]["cyclical"], "operating margin ignores a net-income one-off"
    assert res["impairment"]["margin_basis"] == "operating"
    assert res["micron_fy23"]["yield"] < 0, "a loss-maker stays negative"
    assert math.isnan(res["short"]["yield"]) and res["short"]["blank"] == "insufficient_history"
    bad = dict(_ne_facts("stable"), ne_rev=[143, NA, 198, 212])
    assert M.normalized_earnings(bad)["blank"] == "bad_history"
    assert M.normalized_earnings(dict(_ne_facts("stable"), mcap=NA))["blank"] == "no_market_cap"


def test_v4_wiring_compute_all_financials_and_workings():
    f = dict(_ne_facts("micron_peak"), is_fin=False)
    check("compute_all ranks the normalized yield as V4", M.compute_all(f)["v4_norm_ep"],
          M.normalized_earnings(f)["yield"], 1e-12)
    # financials keep 5y average ROE x book / market cap
    fin = {"is_fin": True, "roe_5y_avg": 0.12, "roe_years": 5, "book": 50.0, "mcap": 100.0}
    check("financial V4 = ROE x book / mcap", M.v4_norm_ep(fin), 0.06, 1e-12)
    # facts carry the annual history oldest first, capped at max_years
    inc_a = [{"date": f"{2025 - i}-12-31", "revenue": 100.0 + i, "netIncome": 10.0,
              "operatingIncome": 12.0} for i in range(9)]
    hist = F.build_facts("H", _blob(income_a=inc_a), {}, {})
    assert len(hist["ne_rev"]) == C.NE_PARAMS["max_years"]
    assert hist["ne_dates"] == sorted(hist["ne_dates"]) and hist["ne_rev"][-1] == 100.0
    # every step shown in the workings reproduces the metric
    w = W.build(f, M.compute_all(f))["v4_norm_ep"]
    assert w["ok"], w


def test_fwd_rev_is_context_with_ntm_rules():
    def fwd(est, ttm=90.0):
        f = F.build_facts("R", _blob(estimates=est), {}, {})
        return f["fwd_rev"], f["fwd_rev_basis"]
    v, b = fwd([_est(182, 100.0), _est(547, 120.0)])
    assert b == "ntm" and 100.0 < v < 120.0
    v, b = fwd([_est(182, 100.0, 2), _est(547, 120.0, 2)])
    assert b == "none" and math.isnan(v), "fewer than MIN_ANALYSTS -> no consensus"
    assert not hasattr(F, "v4_revenue") and not hasattr(C, "NTM_REV_BASIS_FLOOR")


def test_quality_is_one_block_hygiene_not_applicable_to_financials():
    raw = pd.DataFrame({"is_fin": [False, True, False]})
    pcts = pd.DataFrame({"q1_roic": [60.0, 60.0, 60.0], "q2_roiic": [60.0, 60.0, 60.0],
                         "q3_persistence": [60.0, 60.0, 60.0],
                         "q4_leverage_score": [90.0, 90.0, 90.0],
                         "q5_drawdown": [90.0, 90.0, 90.0],
                         "q6_hygiene": [30.0, np.nan, np.nan]})
    p = R.build_pillars(raw, pcts)
    check("operating: mean of six", float(p["quality_raw"].iloc[0]), (180 + 180 + 30) / 6)
    check("financial: mean of five, hygiene left out (not 50)",
          float(p["quality_raw"].iloc[1]), (180 + 180) / 5)
    check("operating with missing hygiene: neutral 50",
          float(p["quality_raw"].iloc[2]), (180 + 180 + 50) / 6)
    check("financial protection: leverage and drawdown only",
          float(p["safety_protection"].iloc[1]), 90.0)
    assert "quality_damp" not in p.columns and not hasattr(C, "SHIELD_FLOOR")
    # financials: hygiene not computed
    m = M.compute_all({"is_fin": True, "shares_a0": 1.1e9, "shares_3y": 1e9,
                       "sbc_ttm": 1e8, "rev_ttm": 1e10})
    assert all(math.isnan(m[k]) for k in ("hyg_accruals", "hyg_dilution", "hyg_bloat"))


def test_hygiene_rollback_restores_neutral_fill(monkeypatch):
    monkeypatch.setattr(C, "FIN_NOT_APPLICABLE", ())
    raw = pd.DataFrame({"is_fin": [True]})
    pcts = pd.DataFrame({k: [60.0] for k, _ in C.QUALITY_METRICS} | {"q6_hygiene": [np.nan]})
    check("rollback: financial hygiene is neutral 50",
          float(R.build_pillars(raw, pcts)["quality_raw"].iloc[0]), (300 + 50) / 6)


def test_p1_p2_use_compounded_window_return():
    r = [0.10, -0.05, 0.02, 0.04, 0.03, -0.01, 0.06, 0.02, 0.01, 0.05, -0.02]
    sd = float(np.std(r, ddof=1))
    check("P1 = (prod(1+r) - 1) / SD", M.p1_trend_12_1({"monthly_returns_12_1": r}),
          (float(np.prod([1 + x for x in r])) - 1) / sd, 1e-12)
    # same mean monthly return, different compounded total -> rank apart
    a, b = [0.10, -0.10, 0.10, -0.10, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]
    b = [0.02, -0.02, 0.02, -0.02, 0.0]
    assert M.p2_trend_6_1({"monthly_returns_6_1": a}) != M.p2_trend_6_1({"monthly_returns_6_1": b})


def test_checklist_quality_rank_and_financials_out_of_nine():
    df = pd.DataFrame({"is_fin": [False, True], f"composite_rank_{C.BASE_PROFILE}": [80.0, 80.0],
                       "grp_base": [70.0, 70.0], "v_rank_s": [70.0, 70.0], "q_rank_u": [75.0, 75.0],
                       "leverage_ok": [100.0, np.nan], "b_rank_u": [75.0, 75.0],
                       "p_rank_u": [75.0, 75.0], "price_credibility": [60.0, 60.0],
                       "q6_hygiene_pct": [50.0, np.nan]})
    out = R.checklist(df)
    assert out["Quality >= 70"].tolist() == ["PASS", "PASS"]
    assert out["Hygiene >= 40"].tolist() == ["PASS", "N/A"]
    assert out["item_count"].tolist() == [10, 9]
    assert out["pass_count"].tolist() == [9, 7]


def test_run_has_no_version_label_and_financial_hygiene_is_blank():
    data, bench = _data()
    for t in ("S03", "S11"):                          # make two names banks
        prof = data[t]["blob"]["profile"][0]
        prof["sector"], prof["industry"] = "Financial Services", "Banks - Regional"
    full = E._run_full(data, bench, workers=1)
    assert "spec_version" not in full.attrs and not hasattr(C, "SPEC_VERSION")
    fin = full["is_fin"].fillna(False).astype(bool)
    if fin.any():
        assert full.loc[fin, "q6_hygiene_pct"].isna().all()
        assert not full.loc[fin, "flag_HYGIENE"].any()
    out = E.to_output(full, C.BASE_PROFILE)
    assert len(E.COLUMNS) == 146 and list(out.columns) == E.COLUMNS
    assert (out["checklist_items"] == np.where(out["fin_mode"] == 1, 9, 10)).all()
    assert not out.loc[out["fin_mode"] == 1, "neutral_fill_metrics"].str.contains("hygiene").any()


def test_sector_lens_pairs_regimes_and_attrs():
    from vqmb import sector_lens as L
    W_ = C.PROFILES["GROWTH"]
    leg = lambda sym, rnk, v, q, b, p, ad, sf, flags=(): {
        "sym": sym, "name": sym, "rnk": rnk, "V": v, "Q": q, "B": b, "P": p,
        "ad": ad, "sf": sf, "dv": "DRIFT", "st": "", "flags": list(flags)}
    shares = lambda parts: "/".join(str(round(parts[k]["share"] * 100)) for k in "BPQV")
    mu, qcom = leg("MU", 100, 66, 97, 99, 77, "A", "D"), leg("QCOM", 8, 81, 33, 7, 34, "C", "D")
    pt = L.decompose(mu, qcom, W_)
    assert shares(pt) == "57/17/25/-2" and max(pt, key=lambda k: pt[k]["contrib"]) == "B"
    assert ("/".join(L._regime(b, p) for b, p in ((60, 60), (60, 40), (40, 60), (40, 40), (50, 60)))
            == "CONFIRMED/TAPE LAGS/TAPE LEADS/BROKEN/MIXED")
    assert "/".join(L._dispersion(v) for v in (41, 26, 12)) == "WIDE/MODERATE/TIGHT"
    assert (L.short_bar(pd.Series([10, 20, 30, 90, 95, 99])), L.short_bar(pd.Series([60, 70, 80, 90]))) == (40, 50)

    rows = []
    def add(sec, sym, rnk, v, q, b, p, **fl):
        rows.append({"symbol": sym, "company": sym, "group": sec,
                     **{f"composite_rank_{pr}": rnk for pr in C.PROFILES},
                     "v_rank_u": v, "q_rank_u": q, "b_rank_u": b, "p_rank_u": p, "ad": "B",
                     "safety": "B", "darvas": "DRIFT", "darvas_stack": "", "gated": False,
                     **{f"flag_{k}": True for k in fl}})
    for i, r in enumerate((95, 92, 90, 88, 86, 84)):
        add("Block", f"B{i}", r, 30, 70, 85, 80)
    for i, (r, b, p) in enumerate(((90, 80, 85), (85, 75, 80), (80, 70, 70), (20, 30, 20),
                                   (15, 20, 30), (10, 25, 10), (50, 50, 50))):
        add("Spread", f"S{i}", r, 50, 50, b, p)
    add("Spread", "S7", 99, 50, 50, 90, 95, STALE=True)
    for i, r in enumerate((60, 30, 20, 10)):
        add("Weak", f"W{i}", r, 70, 40, 30, 20)
    add("Tiny", "T0", 90, 50, 50, 50, 50)
    df = pd.DataFrame(rows)
    secs, pairs, _ = L.sector_lens(df, "GROWTH")
    by = secs.set_index("sector")
    assert (by.loc["Block", "regime"], by.loc["Block", "dispersion"], by.loc["Block", "no_pair"]) == \
        ("CONFIRMED", "TIGHT", "no_short")
    assert by.loc["Weak", "no_pair"] == "no_long" and bool(by.loc["Tiny", "thin"])
    sp = pairs[pairs["sector"] == "Spread"]
    assert len(sp) == 2 and (sp.iloc[0]["long"], sp.iloc[0]["short"]) == ("S0", "S5")
    assert "S7" not in set(sp["long"])
    att = L.lens_attrs(df)
    assert sorted(att) == sorted(C.PROFILES)
    json.dumps(att)                                   # plain records, serialisable
    assert len(att["GROWTH"]["pairs"]) == len(pairs)

    # a real run carries the lens for every profile in attrs
    data, bench = _data()
    full = E._run_full(data, bench, workers=1)
    assert sorted(full.attrs["sector_lens"]) == sorted(C.PROFILES)
    json.dumps(full.attrs["sector_lens"])
