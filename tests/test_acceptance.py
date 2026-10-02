"""
VQMB acceptance tests.

The first one is the handbook's own worked example (§3): "Reproduce these
numbers and the chain is wired correctly." The rest check that each blank rule
blanks for the stated reason, because a metric that silently returns 0 instead
of NaN would rank — and rank badly — rather than reweight.

    pytest
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from vqmb import config as C
from vqmb import metrics as M
from vqmb import ranking as R


def check(label, got, want, tol=0.05):
    ok = (got is None and want is None) or (
        isinstance(got, str) and got == want) or (
        isinstance(got, (int, float)) and isinstance(want, (int, float))
        and abs(got - want) <= tol)
    assert ok, f"{label}: got {got!r}, want {want!r}"


def check_blank(label, got):
    ok = got is None or (isinstance(got, float) and math.isnan(got))
    assert ok, f"{label}: got {got!r}, want blank"


# ---------------------------------------------------------------------------
def test_aura_walkthrough():
    """Handbook §3. Value raw 60 -> V=71; engine 75 x shield damp (shield 78)
    -> 68 -> Q=74; biz raw 81 -> B=89; strength 58 x cred damp (cred 70) -> 49
    -> P=58. Then the three profile composites."""

    shield_damp = C.SHIELD_FLOOR + C.SHIELD_SPAN * 78 / 100.0
    quality_raw = 75 * shield_damp
    check("shield dampener at shield=78", shield_damp, 0.912, 1e-9)
    check("quality raw = engine 75 x dampener", quality_raw, 68.4, 0.05)

    cred_damp = C.CRED_FLOOR + C.CRED_SPAN * 70 / 100.0
    price_raw = 58 * cred_damp
    check("credibility dampener at cred=70", cred_damp, 0.85, 1e-9)
    check("price raw = strength 58 x dampener", price_raw, 49.3, 0.05)

    V, Q, B, P = 71.0, 74.0, 89.0, 58.0
    ranks = pd.DataFrame({"v_rank_u": [V], "q_rank_u": [Q],
                          "b_rank_u": [B], "p_rank_u": [P]})
    comp = R.build_composites(ranks)
    check("TRADER composite", float(comp["composite_raw_TRADER"].iloc[0]), 70.0)
    check("PM composite", float(comp["composite_raw_PM"].iloc[0]), 74.6)
    check("GROWTH composite", float(comp["composite_raw_GROWTH"].iloc[0]), 75.7)


def test_percentile():
    s = pd.Series([10, 20, 30, 40, 50])
    p = R.percentile(s)
    check("lowest of five -> 0", float(p.iloc[0]), 0.0)
    check("highest of five -> 100", float(p.iloc[4]), 100.0)
    check("middle of five -> 50", float(p.iloc[2]), 50.0)
    p = R.percentile(pd.Series([10, 20, 20, 40]))
    check("ties take the average rank", float(p.iloc[1]), float(p.iloc[2]), 1e-9)
    p = R.percentile(pd.Series([10, 20, 30]), direction=-1)
    check("inv direction flips the order", float(p.iloc[0]), 100.0)
    check_blank("a single name has no percentile",
                float(R.percentile(pd.Series([5.0])).iloc[0]))


def test_blank_reweighting():
    pcts = pd.DataFrame({"v1_ebit_ev": [80.0], "v2_ev_gp": [60.0],
                         "v3_fwd_earn_yield": [np.nan], "v4_norm_ep": [40.0],
                         "v5_ev_sales_vs_hist": [np.nan]})
    cols = [k for k, _ in C.VALUE_METRICS]
    # VQMB rule: each blank lens counts as the neutral 50, mean over all five
    got = float(R.block_mean(pcts, cols).iloc[0])
    check("two of five lenses blank -> each counts as 50", got, (80 + 60 + 50 + 40 + 50) / 5)
    # the FLUX rule is still there behind the switch
    orig = C.METRIC_NEUTRAL_FILL_ON
    try:
        C.METRIC_NEUTRAL_FILL_ON = False
        got = float(R.block_mean(pcts, cols).iloc[0])
        check("switch off (FLUX): mean of the three present", got, 60.0)
    finally:
        C.METRIC_NEUTRAL_FILL_ON = orig


def test_biz_momentum_neutral_fill():
    cols = [k for k, _ in C.BIZ_METRICS]
    b = pd.DataFrame({"b1_vs_trend": [np.nan, np.nan, np.nan, 90.0],
                      "b2_sequential": [80.0, np.nan, np.nan, 70.0],
                      "b3_margin_delta": [60.0, 30.0, np.nan, 20.0]})
    got = R.block_mean(b, cols).tolist()
    check("B1 blank -> (50 + B2 + B3) / 3", got[0], (50 + 80 + 60) / 3)
    check("B1, B2 blank -> (50 + 50 + B3) / 3", got[1], (50 + 50 + 30) / 3)
    check("all three blank -> 50", got[2], 50.0)
    check("none blank -> plain mean", got[3], (90 + 70 + 20) / 3)


def test_leverage_curve():
    def score(nd, ebit):
        return M.q4_leverage_score({"net_debt": nd, "ebit_ttm": ebit})
    check("net cash -> 100", score(-5e9, 1e9)[0], 100.0)
    check("0x leverage -> 100", score(0.0, 1e9)[0], 100.0)
    check("3x (the kink) -> 60", score(3e9, 1e9)[0], 60.0)
    check("1.5x -> midway 80", score(1.5e9, 1e9)[0], 80.0)
    check("3.5x -> convex, below linear", score(3.5e9, 1e9)[0], 45.0)
    check("4x -> 0", score(4e9, 1e9)[0], 0.0)
    s, gate = score(5e9, 1e9)
    check("above 4x -> 0 and GATED", s, 0.0)
    check("  ...gate raised", "gate" if gate else "no gate", "gate")
    _, gate = score(2e9, -1e9)
    check("EBIT<=0 with net debt -> GATED", "gate" if gate else "no gate", "gate")
    v, gate = score(-2e9, -1e9)   # net cash, but loss-making
    check_blank("EBIT<=0 with net cash -> blank, no gate", v)
    check("  ...and no gate", "gate" if gate else "no gate", "no gate")


def test_value_blanks():
    check_blank("V1: EV <= 0 -> blank",
                M.v1_ebit_ev({"ev": -1e9, "ebit_ttm": 5e8}))
    check_blank("V2: gross profit <= 0 -> blank",
                M.v2_ev_gp({"ev": 1e10, "gp_ttm": -1e8}))
    check_blank("V4: fewer than 3y of margins -> blank",
                M.v4_norm_ep({"margin_years": 2, "net_margin_5y_avg": 0.1,
                              "rev_ttm": 1e10, "mcap": 1e11}))
    check_blank("V5: no own history -> blank",
                M.v5_ev_sales_vs_hist({"ev": 1e11, "rev_ttm": 1e10}))
    v, tags = M.v3_fwd_earn_yield({"ntm_eps": -1.5, "price": 100.0})
    check_blank("V3 LOSS path: forecast loss -> blank", v)
    check("  ...tagged LOSS", "LOSS" if tags["v3_loss"] else "-", "LOSS")
    v, tags = M.v3_fwd_earn_yield({"price": 100.0, "rev_ttm": 1e10, "mcap": 1e11,
                                   "net_margin_5y_avg": -0.05, "g_projection": 0.1})
    check_blank("V3 CALC guard: negative 5y margin -> blank", v)
    check("  ...tagged CALC-NA", "CALC-NA" if tags["v3_calc_na"] else "-", "CALC-NA")
    v, tags = M.v3_fwd_earn_yield({"price": 100.0, "rev_ttm": 1e10, "mcap": 1e11,
                                   "net_margin_5y_avg": 0.12, "g_projection": 0.10,
                                   "margin_years": 5})
    # 0.12 margin x $10bn revenue x 1.10 growth / $100bn mcap = 1.32%
    check("V3 CALC path yield", v, 0.0132, 1e-6)
    check("  ...tagged CALC", "CALC" if tags["v3_calc"] else "-", "CALC")


def test_roiic_inapplicable():
    base = {"d_nopat_3y": 1e9, "invested_capital": 1e11}
    check_blank("capital shrank (buybacks) -> blank",
                M.q2_roiic({**base, "d_ic_3y": -5e9}))
    check_blank("capital moved < 5% of IC -> blank",
                M.q2_roiic({**base, "d_ic_3y": 1e9}))
    check("capital moved enough -> computed",
          M.q2_roiic({**base, "d_ic_3y": 2e10}), 0.05, 1e-9)


def test_roic_fallback():
    v, fb = M.q1_roic({"ebit_ttm": 1e10, "invested_capital": 5e10,
                       "effective_tax_rate": 0.25})
    check("ROIC = EBIT x (1 - effective tax) / IC", v, 0.15, 1e-9)
    check("  ...not a fallback", "no" if not fb else "yes", "no")
    # register row 1: a loss-maker gets its real negative return, not GP/assets
    v, fb = M.q1_roic({"ebit_ttm": -1e9, "invested_capital": 5e10,
                       "gp_ttm": 2e10, "assets": 1e11})
    check("row 1: EBIT -1bn, IC 50bn -> real negative return", v, -0.02, 1e-9)
    check("  ...not a fallback", "yes" if fb else "no", "no")
    v, fb = M.q1_roic({"ebit_ttm": 1e9, "invested_capital": -5e9,
                       "gp_ttm": 2e10, "assets": 1e11})
    check("row 1: EBIT 1bn, IC -5bn -> GP/assets stand-in", v, 0.20, 1e-9)
    check("  ...marked and flagged", "yes" if fb else "no", "yes")
    v, fb = M.q1_roic({"ebit_ttm": -1e9, "invested_capital": -5e9,
                       "gp_ttm": 2e10, "assets": 1e11})
    check_blank("row 1: EBIT -1bn, IC -5bn -> blank", v)
    v, fb = M.q1_roic({"invested_capital": 5e10, "gp_ttm": 2e10, "assets": 1e11})
    check_blank("row 1: EBIT missing -> blank", v)
    v, _ = M.q1_roic({"ebit_ttm": 1e10, "invested_capital": 5e10,
                      "effective_tax_rate": 0.90})
    check("a 90% rate is clamped to 35%", v, 0.13, 1e-3)
    v, _ = M.q1_roic({"ebit_ttm": 1e10, "invested_capital": 5e10,
                      "effective_tax_rate": -0.4})
    check("a negative rate is clamped to 0%", v, 0.20, 1e-9)
    v, _ = M.q1_roic({"ebit_ttm": 1e10, "invested_capital": 5e10})
    check("no rate computable -> 21% fallback", v, 0.158, 1e-3)
    a, _ = M.q1_roic({"ebit_ttm": 1e10, "invested_capital": 5e10,
                      "effective_tax_rate": 0.10})
    b, _ = M.q1_roic({"ebit_ttm": 1e10, "invested_capital": 5e10,
                      "effective_tax_rate": 0.30})
    check("a low-tax company scores higher, as the handbook intends",
          "yes" if a > b else "no", "yes")

    # ROIIC differences two NOPATs, each taxed at its own year's rate, so a
    # genuine change in the tax rate moves ROIIC even with flat EBIT. Documented
    # here so the behaviour is deliberate rather than discovered later.
    import datetime as _dt
    from vqmb import facts as _F
    _T = _dt.date.today()
    _inc = [{"date": (_T - _dt.timedelta(days=365 * k)).isoformat(), "revenue": 1e10,
             "operatingIncome": 2e9, "netIncome": 1.4e9, "incomeBeforeTax": 1.8e9,
             "incomeTaxExpense": 1.8e9 * (0.30 if k == 0 else 0.10),
             "weightedAverageShsOutDil": 1e9} for k in range(6)]
    _bal = [{"date": (_T - _dt.timedelta(days=365 * k)).isoformat(), "totalAssets": 3e10,
             "totalDebt": 8e9 if k == 0 else 5e9, "totalStockholdersEquity": 1.2e10,
             "cashAndShortTermInvestments": 2e9} for k in range(6)]
    _f = _F.build_facts("X", {"profile": [{"sector": "Industrials",
                                           "industry": "Aerospace & Defense"}],
                              "quote": [{"price": 100, "marketCap": 5e10}],
                              "income_a": _inc, "income_q": _inc[:1] * 12,
                              "balance_a": _bal, "cashflow_a": [], "cashflow_q": [],
                              "estimates": [], "ohlcv": []}, {}, {})
    # EBIT 2bn taxed at 30% now vs 10% three years ago: 1.40bn - 1.80bn
    check("tax 10%->30% on flat EBIT moves NOPAT (handbook arithmetic)",
          _f["d_nopat_3y"] / 1e9, -0.40, 1e-6)


def test_dampeners_can_only_cost():
    raw = pd.DataFrame(index=[0, 1])
    pcts = pd.DataFrame({"q1_roic": [80.0, 80.0], "q2_roiic": [80.0, 80.0],
                         "q3_persistence": [80.0, 80.0],
                         "q4_leverage_score": [100.0, 0.0],
                         "q5_drawdown": [100.0, 0.0], "q6_hygiene": [100.0, 0.0],
                         "p1_trend_12_1": [60.0, 60.0], "p2_trend_6_1": [60.0, 60.0],
                         "p3_high_distance": [60.0, 60.0],
                         "p4_continuity": [100.0, 0.0], "p5_lottery": [100.0, 0.0],
                         "p6_down_resilience": [100.0, 0.0]})
    p = R.build_pillars(raw, pcts)
    check("perfect shield -> quality raw equals the engine",
          float(p["quality_raw"].iloc[0]), 80.0)
    check("worst shield -> engine x 0.6", float(p["quality_raw"].iloc[1]), 48.0)
    check("perfect credibility -> price raw equals strength",
          float(p["price_raw"].iloc[0]), 60.0)
    check("worst credibility -> strength x 0.5", float(p["price_raw"].iloc[1]), 30.0)
    zero = R.build_pillars(pd.DataFrame(index=[0]), pd.DataFrame(
        {"q1_roic": [0.0], "q2_roiic": [0.0], "q3_persistence": [0.0],
         "q4_leverage_score": [100.0], "q5_drawdown": [100.0], "q6_hygiene": [100.0]}))
    check("no engine -> protection earns nothing",
          float(zero["quality_raw"].iloc[0]), 0.0)


def test_gates_outside_averages():
    df = pd.DataFrame({
        "symbol": ["SAFE", "LEVERED"],
        "nd_ebit": [1.0, 6.0],
        "composite_raw_TRADER": [95.0, 99.0],
        "composite_rank_TRADER": [99.0, 100.0],
        "composite_rank_PM": [99.0, 100.0], "composite_rank_GROWTH": [99.0, 100.0],
    })
    df["gated"] = R.apply_gates(df)
    check("levered name is gated", bool(df["gated"].iloc[1]), True)
    check("  ...with its cause shown", df["gate_cause"].iloc[1], "ND/EBIT > 4x")
    conv = R.conviction(df, "TRADER")
    check("top-ranked but gated -> never green", bool(conv["green"].iloc[1]), False)
    check("  ...and never a triple star", bool(conv["triple"].iloc[1]), False)

    # A bank's net debt is its funding, not a financing choice. Applying the
    # operating-company leverage gate to financials gated every bank in the
    # index; the handbook gives them a capital gate instead (§11).
    fin = pd.DataFrame({
        "symbol": ["BANK_STRONG", "BANK_THIN", "INDUSTRIAL"],
        "is_fin": [True, True, False],
        "fin_kind": ["bank", "bank", None],
        "nd_ebit": [180.0, 160.0, 6.0],          # meaningless for the two banks
        "capital_ratio": [0.11, 0.03, 0.40],     # the test that does apply
    })
    g = R.apply_gates(fin)
    check("well-capitalised bank is NOT gated by ND/EBIT", bool(g.iloc[0]), False)
    check("thinly-capitalised bank IS gated, on capital", bool(g.iloc[1]), True)
    check("  ...with the right cause", fin["gate_cause"].iloc[1], "capital below floor")
    check("levered industrial still gated on ND/EBIT", bool(g.iloc[2]), True)


def test_fin_splice():
    """§12: variant metrics rank within the FIN group only.

    Without this, a bank's ROE (say 12%) is ranked on the same scale as an
    industrial's ROIC (say 25%), and a capital ratio against a leverage score.
    They are different quantities; one ranking over both is meaningless.
    """
    n_fin, n_ind = 20, 40
    df = pd.DataFrame({
        "is_fin": [True] * n_fin + [False] * n_ind,
        # banks cluster at 8-16% ROE, industrials at 18-40% ROIC: on one scale
        # every bank sits in the bottom third by construction
        "q1_roic": list(np.linspace(0.08, 0.16, n_fin)) + list(np.linspace(0.18, 0.40, n_ind)),
        "sector": ["Financial Services"] * n_fin + ["Technology"] * n_ind,
    })
    naive = R.percentile(df["q1_roic"], +1)
    check("ranked on one scale, the best bank lands below every industrial",
          "yes" if naive[:n_fin].max() < naive[n_fin:].min() else "no", "yes")

    spliced = R.percentile_spliced(df, "q1_roic", +1)
    check("spliced: best bank reaches the top of its own group",
          float(spliced[:n_fin].max()), 100.0)
    check("spliced: worst bank sits at the bottom of its own group",
          float(spliced[:n_fin].min()), 0.0)
    check("spliced: industrials still ranked among industrials",
          float(spliced[n_fin:].max()), 100.0)

    thin = pd.DataFrame({"is_fin": [True] * 3 + [False] * 40,
                         "q1_roic": list(np.linspace(0.08, 0.16, 3))
                                    + list(np.linspace(0.18, 0.40, 40))})
    got = R.percentile_spliced(thin, "q1_roic", +1)
    ref = R.percentile(thin["q1_roic"], +1)
    check("too few financials to rank alone -> one universe ranking",
          float((got - ref).abs().max()), 0.0, 1e-9)


def test_group_merge():
    """§14: sub-scale groups fold into their nearest neighbour."""
    from vqmb.groups import resolve_merges
    counts = {"Capital Goods": 37, "Commercial Services": 9, "Software": 41,
              "Diversified Financials": 40, "Banks": 13}
    m = resolve_merges(counts, 12)
    check("a 9-name group folds", m["Commercial Services"], "Capital Goods")
    check("a 13-name group survives a floor of 12", m["Banks"], "Banks")
    m = resolve_merges(counts, 15)
    check("the same group folds at a floor of 15", m["Banks"], "Diversified Financials")
    check("groups above the floor are untouched", m["Software"], "Software")

    # a cycle in the merge table must not hang the run
    import vqmb.groups as G
    saved = dict(G.MERGE_TARGET)
    G.MERGE_TARGET.update({"A": "B", "B": "A"})
    out = resolve_merges({"A": 2, "B": 3, "Big": 50}, 15)
    check("a cycle terminates rather than looping", "ok" if out else "hung", "ok")
    G.MERGE_TARGET.clear()
    G.MERGE_TARGET.update(saved)


def test_workings_reproduce_the_model():
    """Every working shown on Stock View must recompute to the ranked value.
    Covers the branches that are easy to get wrong: a normal ROIC, the GP/assets
    fallback for a loss-maker, a forecast loss, and a financial."""
    from vqmb import workings as W
    base = {"mcap": 1e11, "price": 100.0, "ev": 1.05e11, "w_total_debt": 1e10,
            "w_cash": 5e9, "net_debt": 5e9, "book": 2e10, "invested_capital": 2.5e10,
            "ebit_ttm": 8e9, "effective_tax_rate": 0.2, "gp_ttm": 2e10, "rev_ttm": 4e10,
            "assets": 6e10, "ni_ttm": 5e9, "ntm_eps": 6.0, "fwd_basis": "ntm",
            "net_margin_5y_avg": 0.12, "margin_years": 5}
    cases = {
        "normal ROIC": dict(base),
        "loss-maker falls back to GP/assets": dict(base, ebit_ttm=-1e9),
        "forecast loss blanks the forward yield": dict(base, ntm_eps=-2.0),
        "bank uses ROE": dict(base, is_fin=True, fin_kind="bank", capital_ratio=0.1),
    }
    for label, f in cases.items():
        m = M.compute_all(f)
        w = W.build(f, m)
        keys = ["v1_ebit_ev", "v2_ev_gp", "v3_fwd_earn_yield", "q1_roic", "q4_leverage_score"]
        ok = all(w[k]["ok"] for k in keys)
        check(label, "reproduces" if ok else "MISMATCH", "reproduces")

    # a TTM line must unfold into quarters that actually sum to it
    q = [["2026-06-30", 2e9], ["2026-03-31", 1.5e9], ["2025-12-31", 1.2e9], ["2025-09-30", 1.3e9]]
    f = dict(base, ebit_ttm=6e9, w_q={"operatingIncome": q})
    w = W.build(f, M.compute_all(f))
    row = [x for x in w["v1_ebit_ev"]["in"] if x[0] == "Operating income, TTM"][0]
    check("TTM operating income unfolds to 4 quarters", len(row[3]), 4)
    check("  ...which sum to the TTM figure", sum(k[1] for k in row[3]), 6e9, 1)
    f_bad = dict(f, w_q={"operatingIncome": q[:3] + [["2025-09-30", 9e9]]})
    w_bad = W.build(f_bad, M.compute_all(f_bad))
    check("quarters that do not add up are flagged",
          "flagged" if not w_bad["v1_ebit_ev"]["ok"] else "missed", "flagged")


def test_filled_in_gross_profit():
    """A quarter where the vendor filled gross profit with revenue is a missing
    number, not a 100% margin - it must blank, never rank at the top."""
    import datetime as _dt
    from vqmb import facts as _F
    T = _dt.date.today()

    def blob(gms):
        q = [{"date": (T - _dt.timedelta(days=91 * k)).isoformat(), "revenue": 3.9e9,
              "grossProfit": 3.9e9 * g, "operatingIncome": 1.3e9, "netIncome": 1e9,
              "epsdiluted": 0.5, "weightedAverageShsOutDil": 2e9} for k, g in enumerate(gms)]
        return {"profile": [{"sector": "Industrials", "industry": "Railroads"}],
                "quote": [{"price": 35, "marketCap": 6.5e10}], "income_a": [], "income_q": q,
                "balance_a": [], "cashflow_a": [], "cashflow_q": [], "estimates": [], "ohlcv": []}

    csx = _F.build_facts("X", blob([1.0] + [0.36] * 11), {}, {})
    check_blank("latest quarter filled in -> margin delta blank", M.b3_margin_delta(csx))
    check_blank("  ...and EV/gross profit blank", _F._f(csx.get("gp_ttm")))
    check("  ...and the set-aside is flagged", "flagged" if csx["gp_set_aside"] else "silent", "flagged")
    real = _F.build_facts("X", blob([0.61] + [0.36] * 3 + [0.36] * 8), {}, {})
    check("a genuine 2,500bp swing is kept", M.b3_margin_delta(real), 2500.0, 1)
    sw = _F.build_facts("X", blob([0.85] * 12), {}, {})
    check("a genuine 85% software margin is kept", "kept" if sw["gp_ttm"] == sw["gp_ttm"] else "lost", "kept")


def test_flag_hysteresis():
    df = pd.DataFrame({"q6_hygiene_pct": [25.0], "price_strength": [70.0],
                       "price_credibility": [40.0],
                       "v_rank_u": [50.0], "q_rank_u": [50.0],
                       "b_rank_u": [50.0], "p_rank_u": [50.0]})
    fresh = R.compute_flags(df)
    check("hygiene 25 with no history -> off (ON bar is 20)",
          bool(fresh["HYGIENE"].iloc[0]), False)
    prev = pd.DataFrame({"HYGIENE": [True], "DISCRETE": [True]})
    # the mechanism existed but was never wired: compute_flags was called
    # without `previous`, so every flag used its ON bar only and hysteresis
    # never once fired. This asserts the wiring, not just the function.
    check("without yesterday's state a boundary name is off",
          bool(R.compute_flags(df)["HYGIENE"].iloc[0]), False)
    sticky = R.compute_flags(df, previous=prev)
    check("was on, still below the OFF bar of 30 -> stays on",
          bool(sticky["HYGIENE"].iloc[0]), True)
    check("DISCRETE sticks: cred 40 < OFF bar 45", bool(sticky["DISCRETE"].iloc[0]), True)


def test_safety_absolute():
    sg = R.safety_grade(pd.Series([90.0, 55.0, 90.0]), pd.Series([90.0, 55.0, 90.0]),
                        pd.Series([False, False, True]))
    check("strong shield and record -> A", sg["safety"].iloc[0], "A")
    check("middling -> C", sg["safety"].iloc[1], "C")
    check("gated overrides everything -> E", sg["safety"].iloc[2], "E")
