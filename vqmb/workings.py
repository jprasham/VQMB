"""
VQMB — the workings.

For every ranked metric, the raw vendor inputs it was built from, the formula,
and the result. Shown on the Stock View so any number can be traced back to the
statement line it came from.

Every entry recomputes its result from the inputs it displays and compares that
against the value the model actually ranked. If they ever disagree the entry is
marked, rather than quietly showing arithmetic that does not match the score -
the display must never be a paraphrase of the model.

Each entry:
    {"f": formula, "in": [[label, value, kind], ...],
     "out": [label, value, kind], "note": str, "ok": bool | None}

kind: "$" money, "%" percent, "x" multiple, "n" plain number, "pp" percentage
points, "bp" basis points, "d" count, "t" text.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from .metrics import effective_tax

NA = float("nan")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NA
    return v if math.isfinite(v) else NA


def _ok(*v):
    return all(math.isfinite(x) for x in v)


def _close(a, b, tol=1e-6):
    a, b = _f(a), _f(b)
    if not _ok(a) and not _ok(b):
        return True                      # both blank - consistent
    if not _ok(a, b):
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def _conv(v, k):
    if k == "t":
        return v
    return None if not _ok(_f(v)) else float(v)


def _entry(formula, inputs, out_label, result, kind, model_value, note=""):
    return {"f": formula,
            "in": [[row[0], _conv(row[1], row[2]), row[2]] + (row[3:4] if len(row) > 3 else [])
                   for row in inputs],
            "out": [out_label, None if not _ok(_f(result)) else float(result), kind],
            "note": note,
            "ok": _close(result, model_value)}


def _date_note(d):
    return f"statement dated {d}" if d else ""


# ===========================================================================
def build(f: dict, m: dict) -> dict:
    """Workings for all twenty metrics of one name."""
    w: dict = {}
    fin = bool(f.get("is_fin"))
    kind_fin = f.get("fin_kind")
    mcap, price = _f(f.get("mcap")), _f(f.get("price"))
    debt, cash = _f(f.get("w_total_debt")), _f(f.get("w_cash"))
    ev = _f(f.get("ev"))
    ebit_ttm = _f(f.get("ebit_ttm"))
    q_dates = f.get("w_ttm_dates") or []
    ttm_span = (f"{q_dates[-1]} to {q_dates[0]}" if q_dates and all(q_dates)
                else "the last twelve months")
    if f.get("cadence") == "half-yearly":
        ttm_span += " (two half-year reports)"

    ev_inputs = [["Market cap", mcap, "$"], ["Total debt", debt, "$"],
                 ["Cash & short-term investments", cash, "$"],
                 ["Enterprise value = mcap + debt − cash", ev, "$"]]

    # ---------------- VALUE ----------------
    if fin:
        ni = _f(f.get("ni_ttm"))
        r = ni / mcap if _ok(ni, mcap) and mcap > 0 else NA
        w["v1_ebit_ev"] = _entry("TTM net income ÷ market cap",
            [["Net income, TTM", ni, "$"], ["Market cap", mcap, "$"]],
            "Earnings yield", r, "%", m.get("v1_ebit_ev"),
            "financials variant: a lender has no meaningful enterprise value")
    else:
        r = ebit_ttm / ev if _ok(ebit_ttm, ev) and ev > 0 else NA
        w["v1_ebit_ev"] = _entry("TTM operating income ÷ enterprise value",
            [["Operating income, TTM", ebit_ttm, "$"]] + ev_inputs,
            "EBIT / EV", r, "%", m.get("v1_ebit_ev"),
            f"TTM = the four quarters {ttm_span}" + ("; blank because EV ≤ 0" if _ok(ev) and ev <= 0 else ""))

    if fin:
        book = _f(f.get("book_tangible") if kind_fin == "insurer" else f.get("book"))
        r = book / mcap if _ok(book, mcap) and mcap > 0 else NA
        lab = "Tangible book" if kind_fin == "insurer" else "Book equity"
        w["v2_ev_gp"] = _entry(f"{lab.lower()} ÷ market cap",
            [[lab, book, "$"], ["Market cap", mcap, "$"]],
            "Book / price", r, "x", m.get("v2_ev_gp"),
            "financials variant; higher is cheaper here, the opposite of EV/GP")
    else:
        gp = _f(f.get("gp_ttm"))
        r = ev / gp if _ok(ev, gp) and gp > 0 and ev > 0 else NA
        note = "lower is cheaper"
        if f.get("gp_quality") in ("none", "mixed") and not _ok(gp):
            note = ("set aside: the vendor reports no cost of revenue for "
                    + ("any quarter" if f.get("gp_quality") == "none" else "some quarters")
                    + ", so gross profit equals revenue there - a filled-in total, not a margin. "
                    "The other four value lenses carry the weight.")
        elif _ok(gp) and gp <= 0:
            note += "; blank because gross profit ≤ 0"
        elif not _ok(ev) or ev <= 0:
            note = "blank because EV ≤ 0 or missing - a negative EV/GP would read as the cheapest name"
        w["v2_ev_gp"] = _entry("enterprise value ÷ TTM gross profit",
            ev_inputs + [["Gross profit, TTM", gp, "$"]],
            "EV / gross profit", r, "x", m.get("v2_ev_gp"), note)

    eps, cov = _f(f.get("ntm_eps")), _f(f.get("ntm_coverage"))
    basis = str(f.get("fwd_basis") or "")
    if _ok(eps) and eps <= 0:
        w["v3_fwd_earn_yield"] = _entry("consensus next-twelve-month EPS ÷ price",
            [["Consensus NTM EPS", eps, "n"], ["Price", price, "$"]],
            "Forward earnings yield", NA, "%", m.get("v3_fwd_earn_yield"),
            "blank: consensus forecasts a loss, and history is never substituted for a forecast loss")
    elif _ok(eps):
        r = eps / price if _ok(price) and price > 0 else NA
        blend = f.get("ntm_blend") or ""
        w["v3_fwd_earn_yield"] = _entry("consensus next-twelve-month EPS ÷ price",
            [["Consensus NTM EPS", eps, "n"], ["Fiscal-year blend", blend, "t"],
             ["Coverage of the next 12 months", cov, "%"], ["Price", price, "$"]],
            "Forward earnings yield", r, "%", m.get("v3_fwd_earn_yield"),
            "fiscal-year estimates blended by how much of each falls inside the next twelve months"
            if basis == "ntm" else "estimate rows cover under 80% of the year: nearest unclosed fiscal year used")
    else:
        g = _f(f.get("g_projection"))          # row 14: missing growth stays missing
        calc_note = ("blank (CALC-NA): no consensus, and the model-built yield could not be "
                     "computed either - missing growth, or fewer than "
                     f"{C.MIN_AVG_YEARS} years behind the five-year average")
        if fin:
            roe5, book = _f(f.get("roe_5y_avg")), _f(f.get("book"))
            r = (roe5 * book * (1 + g) / mcap if _ok(roe5, book, mcap, g) and roe5 > 0 and mcap > 0
                 and _f(f.get("roe_years")) >= C.MIN_AVG_YEARS else NA)
            w["v3_fwd_earn_yield"] = _entry("5y avg ROE × book × (1 + g) ÷ market cap",
                [["5-year average ROE", roe5, "%"], ["Book equity", book, "$"],
                 ["Projection growth g", g, "%"], ["Market cap", mcap, "$"]],
                "Forward earnings yield (modelled)", r, "%", m.get("v3_fwd_earn_yield"),
                "no consensus: model-built" if _ok(r) else calc_note)
        else:
            m5, rev = _f(f.get("net_margin_5y_avg")), _f(f.get("rev_ttm"))
            r = (m5 * rev * (1 + g) / mcap if _ok(m5, rev, mcap, g) and m5 > 0 and mcap > 0
                 and _f(f.get("margin_years")) >= C.MIN_AVG_YEARS else NA)
            w["v3_fwd_earn_yield"] = _entry("TTM revenue × (1 + g) × 5y avg net margin ÷ market cap",
                [["Revenue, TTM", rev, "$"], ["Projection growth g", g, "%"],
                 ["5-year average net margin", m5, "%"], ["Market cap", mcap, "$"]],
                "Forward earnings yield (modelled)", r, "%", m.get("v3_fwd_earn_yield"),
                ("no consensus: model-built. g = delivered TTM revenue growth, floored 0%, "
                 "capped 20% unless tech/communications or growth in 5 of the last 8 quarters")
                if _ok(r) else calc_note)

    if fin:
        roe5, book = _f(f.get("roe_5y_avg")), _f(f.get("book"))
        r = (roe5 * book / mcap if _ok(roe5, book, mcap) and mcap > 0
             and _f(f.get("roe_years")) >= C.MIN_AVG_YEARS else NA)
        w["v4_norm_ep"] = _entry("5y avg ROE × book ÷ market cap",
            [["Annual ROE, newest first", ", ".join(f"{x*100:.1f}%" for x in (f.get("w_roes") or [])), "t"],
             ["5-year average ROE", roe5, "%"], ["Book equity", book, "$"], ["Market cap", mcap, "$"]],
            "Normalized E/P", r, "%", m.get("v4_norm_ep"))
    else:
        m5, rev = _f(f.get("net_margin_5y_avg")), _f(f.get("rev_ttm"))
        r = (m5 * rev / mcap if _ok(m5, rev, mcap) and mcap > 0
             and _f(f.get("margin_years")) >= C.MIN_AVG_YEARS else NA)
        w["v4_norm_ep"] = _entry("5y avg net margin × TTM revenue ÷ market cap",
            [["Annual net margins, newest first",
              ", ".join(f"{x*100:.1f}%" for x in (f.get("w_margins") or [])), "t"],
             ["5-year average net margin", m5, "%"], ["Revenue, TTM", rev, "$"],
             ["Market cap", mcap, "$"]],
            "Normalized E/P", r, "%", m.get("v4_norm_ep"),
            "mid-cycle margins on today's revenue; blank with fewer than 3 years of margins")

    if fin:
        book = _f(f.get("book"))
        cur = mcap / book if _ok(mcap, book) and book > 0 else NA
        med = _f(f.get("pb_median_hist"))
        r = cur / med - 1 if _ok(cur, med) and med > 0 else NA
        w["v5_ev_sales_vs_hist"] = _entry("(current P/B ÷ own 5-year median P/B) − 1",
            [["Market cap", mcap, "$"], ["Book equity", book, "$"],
             ["Current P/B", cur, "x"], ["Own 5-year median P/B", med, "x"]],
            "Premium to own history", r, "%", m.get("v5_ev_sales_vs_hist"))
    else:
        rev = _f(f.get("rev_ttm"))
        cur = ev / rev if _ok(ev, rev) and rev > 0 and ev > 0 else NA
        med = _f(f.get("ev_sales_median_hist"))
        r = cur / med - 1 if _ok(cur, med) and med > 0 else NA
        w["v5_ev_sales_vs_hist"] = _entry("(current EV/sales ÷ own 5-year median EV/sales) − 1",
            [["Enterprise value", ev, "$"], ["Revenue, TTM", rev, "$"],
             ["Current EV/sales", cur, "x"], ["Own 5-year median EV/sales", med, "x"]],
            "Premium to own history", r, "%", m.get("v5_ev_sales_vs_hist"),
            "lower is cheaper; median taken at each of the last five annual statement dates")

    # ---------------- QUALITY - ENGINE ----------------
    if fin:
        ni, book = _f(f.get("ni_ttm")), _f(f.get("book"))
        r = ni / book if _ok(ni, book) and book > 0 else NA
        w["q1_roic"] = _entry("TTM net income ÷ book equity (ROE)",
            [["Net income, TTM", ni, "$"], ["Book equity", book, "$"]],
            "ROE", r, "%", m.get("q1_roic"), "financials variant")
    else:
        eq = _f(f.get("book"))
        nd = _f(f.get("net_debt"))
        ic = _f(f.get("invested_capital"))
        raw_rate = _f(f.get("effective_tax_rate"))
        rate = effective_tax(raw_rate)
        if _ok(ebit_ttm, ic) and ebit_ttm > 0 and ic > 0:
            nopat = ebit_ttm * (1 - rate)
            r = nopat / ic
            w["q1_roic"] = _entry(
                "TTM operating income × (1 − effective tax, clamped 0–35%) ÷ (equity + net debt)",
                [["Operating income, TTM", ebit_ttm, "$"],
                 ["Income tax expense (FY)", _f(f.get("w_tax_exp")), "$"],
                 ["Pre-tax income (FY)", _f(f.get("w_pretax")), "$"],
                 ["Effective tax rate = tax ÷ pre-tax", raw_rate, "%"],
                 ["Tax rate used (clamped 0–35%)", rate, "%"],
                 ["NOPAT = operating income × (1 − rate)", nopat, "$"],
                 ["Total equity", eq, "$"], ["Total debt", debt, "$"],
                 ["Cash & short-term investments", cash, "$"],
                 ["Net debt = debt − cash", nd, "$"],
                 ["Invested capital = equity + net debt", ic, "$"]],
                "ROIC", r, "%", m.get("q1_roic"),
                f"operating income is trailing twelve months, {ttm_span}; tax and balance sheet "
                f"from the latest annual statement ({f.get('w_bal_date') or '—'})"
                + ("; 21% used because no rate could be computed" if not _ok(raw_rate) else ""))
        else:
            gp, assets = _f(f.get("gp_ttm")), _f(f.get("assets"))
            base = [["Operating income, TTM", ebit_ttm, "$"], ["Total equity", eq, "$"],
                    ["Net debt", nd, "$"], ["Invested capital = equity + net debt", ic, "$"]]
            legacy = C.ROIC_LOSS_MODE == "fallback"
            if not legacy and _ok(ebit_ttm, ic) and ebit_ttm <= 0 and ic > 0:
                # register row 1: a loss-maker's real, negative return
                r = ebit_ttm / ic
                w["q1_roic"] = _entry("TTM operating income ÷ invested capital (loss-maker)",
                    base, "ROIC", r, "%", m.get("q1_roic"),
                    "operating income is negative, so this is a real negative return on the "
                    "capital - with no tax credit applied to a loss")
            elif legacy or (_ok(ebit_ttm, ic) and ebit_ttm > 0 and ic <= 0):
                r = gp / assets if _ok(gp, assets) and assets > 0 else NA
                w["q1_roic"] = _entry("TTM gross profit ÷ total assets (stand-in)",
                    base + [["Gross profit, TTM", gp, "$"], ["Total assets", assets, "$"]],
                    "Gross profit / assets", r, "%", m.get("q1_roic"),
                    "invested capital is zero or negative, so ROIC is undefined and gross "
                    "profit ÷ total assets stands in - marked GP/A")
            else:
                w["q1_roic"] = _entry("ROIC", base, "ROIC", NA, "%", m.get("q1_roic"),
                    "blank: operating income or invested capital is missing, or both are "
                    "negative - the other quality-engine metrics carry the weight")

    if fin:
        ni0, ni3 = _f(f.get("w_ni_0")), _f(f.get("w_ni_3"))
        b0, b3 = _f(f.get("w_book_0")), _f(f.get("w_book_3"))
        dn, db = ni0 - ni3 if _ok(ni0, ni3) else NA, b0 - b3 if _ok(b0, b3) else NA
        r = dn / db if _ok(dn, db) and db > 0 else NA
        w["q2_roiic"] = _entry("(net income now − 3y ago) ÷ (book now − 3y ago)",
            [["Net income, latest FY", ni0, "$"], ["Net income, 3 years earlier", ni3, "$"],
             ["Change in net income", dn, "$"], ["Book, latest FY", b0, "$"],
             ["Book, 3 years earlier", b3, "$"], ["Change in book", db, "$"]],
            "Incremental ROE", r, "%", m.get("q2_roiic"))
    else:
        n0, n3 = _f(f.get("w_nopat_0")), _f(f.get("w_nopat_3"))
        i0, i3 = _f(f.get("w_ic_0")), _f(f.get("w_ic_3"))
        dn = n0 - n3 if _ok(n0, n3) else NA
        di = i0 - i3 if _ok(i0, i3) else NA
        ic = _f(f.get("invested_capital"))
        applicable = _ok(dn, di, ic) and ic > 0 and di > 0 and di >= C.ROIIC_MIN_DELTA_IC * ic
        r = dn / di if applicable else NA
        why = ""
        if _ok(di, ic) and not applicable:
            why = ("; blank: capital shrank (buybacks), so there is no incremental return to measure"
                   if di <= 0 else f"; blank: capital moved less than {C.ROIIC_MIN_DELTA_IC:.0%} of invested capital")
        w["q2_roiic"] = _entry("(NOPAT now − NOPAT 3y ago) ÷ (invested capital now − 3y ago)",
            [[f"Operating income, FY {f.get('w_date_a0') or ''}".strip(), _f(f.get("w_ebit_a0")), "$"],
             ["Tax rate that year", _f(f.get("w_rate_a0")), "%"],
             ["NOPAT, latest FY", n0, "$"],
             [f"Operating income, FY {f.get('w_date_a3') or ''}".strip(), _f(f.get("w_ebit_a3")), "$"],
             ["Tax rate that year", _f(f.get("w_rate_a3")), "%"],
             ["NOPAT, 3 years earlier", n3, "$"], ["Change in NOPAT", dn, "$"],
             ["Invested capital, latest FY", i0, "$"], ["Invested capital, 3 years earlier", i3, "$"],
             ["Change in invested capital", di, "$"]],
            "ROIIC", r, "%", m.get("q2_roiic"),
            "each year taxed at its own effective rate" + why)

    hits, qs = _f(f.get("persistence_hits")), _f(f.get("persistence_quarters"))
    need = f.get("persistence_min", C.PERSISTENCE_MIN_QUARTERS)
    LAG = int(f.get("w_lag") or 4)
    unit = "half-year" if LAG == 2 else "quarter"
    r = hits / qs if _ok(hits, qs) and qs >= need and qs > 0 else NA
    yoy = f.get("w_vol_yoy") or []
    q12 = f.get("w_vol_q12") or []
    kids = []
    for i in range(2 * LAG):
        if i + LAG < len(q12):
            d, v = q12[i]
            _, v4 = q12[i + LAG]
            y = yoy[i] if i < len(yoy) else NA
            kids.append([f"{unit} ending {d or '—'} · {_money(v)} vs {_money(v4)} a year earlier",
                         _conv(y, "%"), "%"])
    w["q3_persistence"] = _entry(f"{unit}s with positive YoY growth ÷ {unit}s measured",
        [[f"Year-on-year growth, newest {unit} first", None, "t", kids],
         [f"{unit.capitalize()}s with positive growth", hits, "d"],
         [f"{unit.capitalize()}s measured", qs, "d"]],
        "Growth persistence", r, "%", m.get("q3_persistence"),
        f"measured on {f.get('w_vol_label', 'revenue')}; at least {need} {unit}s required")

    # ---------------- QUALITY - SHIELD ----------------
    if fin:
        eq, assets = _f(f.get("book")), _f(f.get("assets"))
        cap = _f(f.get("capital_ratio"))
        floor = C.INSURER_CAPITAL_GATE if kind_fin == "insurer" else C.BANK_CAPITAL_GATE
        if not _ok(cap):
            r = NA
        elif cap < floor:
            r = 0.0
        else:
            r = float(np.clip((cap - floor) / (2 * floor) * 100.0, 0, 100))
        w["q4_leverage_score"] = _entry("score from equity ÷ assets against the capital floor",
            [["Total equity", eq, "$"], ["Total assets", assets, "$"],
             ["Equity / assets", cap, "%"], ["Capital floor", floor, "%"]],
            "Leverage score", r, "n", m.get("q4_leverage_score"),
            "0 at the floor, 100 at three times it; below the floor the name is gated")
    else:
        nd = _f(f.get("net_debt"))
        inputs = [["Total debt", debt, "$"], ["Cash & short-term investments", cash, "$"],
                  ["Net debt = debt − cash", nd, "$"], ["Operating income, TTM", ebit_ttm, "$"]]
        if not _ok(ebit_ttm):
            # row 11: missing operating profit is missing data, not a loss
            r = NA
            note = ("gated: operating income missing with net debt"
                    if C.MISSING_EBIT_GATES and _ok(nd) and nd > 0
                    else "blank: operating income not reported - missing data is not a loss, "
                         "so there is no gate")
        elif ebit_ttm <= 0:
            r, note = NA, ("gated: operating income ≤ 0 with net debt" if _ok(nd) and nd > 0
                           else "blank: operating income ≤ 0 with net cash")
        elif _ok(nd) and nd <= 0:
            r, note = 100.0, "net cash scores 100"
        else:
            L = nd / ebit_ttm
            inputs.append(["Net debt / EBIT", L, "x"])
            if L > C.LEV_MAX:
                r, note = 0.0, f"above {C.LEV_MAX:g}×: scores 0 and the name is gated"
            elif L <= C.LEV_KINK:
                r = 100.0 - (100.0 - C.LEV_SCORE_AT_KINK) * (L / C.LEV_KINK)
                note = f"0–{C.LEV_KINK:g}×: falls linearly 100 → {C.LEV_SCORE_AT_KINK:g}"
            else:
                r = C.LEV_SCORE_AT_KINK * (1 - (L - C.LEV_KINK) ** 2)
                note = (f"{C.LEV_KINK:g}–{C.LEV_MAX:g}×: convex, "
                        f"{C.LEV_SCORE_AT_KINK:g} × (1 − (L − {C.LEV_KINK:g})²)")
        w["q4_leverage_score"] = _entry("score from net debt ÷ TTM operating income", inputs,
            "Leverage score", r, "n", m.get("q4_leverage_score"), note)

    n, worst, yrs = _f(f.get("dd_episodes")), _f(f.get("dd_worst")), _f(f.get("price_years"))
    if not _ok(n, yrs) or yrs < C.DRAWDOWN_MIN_YEARS:
        r = NA
    else:
        r = n + abs(worst if _ok(worst) else 0.0) * 100.0 / 50.0
    w["q5_drawdown"] = _entry("episode count + |worst drawdown| ÷ 50",
        [["Years of price history", yrs, "n"],
         [f"Episodes of {C.DRAWDOWN_EPISODE:.0%} or worse", n, "d"],
         ["Worst drawdown", worst, "%"]],
        "Drawdown score", r, "n", m.get("q5_drawdown"),
        f"lower is better; an episode opens at {C.DRAWDOWN_EPISODE:.0%} from a peak and closes on "
        f"recovery to within {1 - C.DRAWDOWN_RECOVERY:.0%} of it")

    ni, cfo, assets = _f(f.get("ni_ttm")), _f(f.get("cfo_ttm")), _f(f.get("assets"))
    rev, rev3 = _f(f.get("rev_ttm")), _f(f.get("rev_3y"))
    rev0 = _f(f.get("rev_a0"))                      # row 29: annual FY0, not TTM
    a3 = _f(f.get("assets_3y"))
    sn, s3 = _f(f.get("shares_a0")), _f(f.get("shares_3y"))   # row 29: annual diluted shares
    fy0, fy3 = f.get("w_date_bal0") or "FY0", f.get("w_date_bal3") or "FY-3"
    sbc = _f(f.get("sbc_ttm"))
    acc, dil, blo = _f(m.get("hyg_accruals")), _f(m.get("hyg_dilution")), _f(m.get("hyg_bloat"))
    pa, pd_, pb = _f(m.get("hyg_accruals_pct")), _f(m.get("hyg_dilution_pct")), _f(m.get("hyg_bloat_pct"))
    parts = [x for x in (pa, pd_, pb) if _ok(x)]
    r = float(np.mean(parts)) if parts else NA
    w["q6_hygiene"] = _entry("mean of three percentiles: accruals, dilution, bloat",
        [["Net income, TTM", ni, "$"], ["Operating cash flow, TTM", cfo, "$"],
         ["Total assets", assets, "$"],
         ["Accruals = (net income − operating cash) ÷ assets", acc, "%"],
         ["  → percentile (lower accruals better)", pa, "n"],
         [f"Diluted shares, FY {fy0}", sn, "n"], [f"Diluted shares, FY {fy3}", s3, "n"],
         ["Stock comp, TTM", sbc, "$"], ["Revenue, TTM", rev, "$"],
         ["Dilution = share CAGR + stock comp ÷ revenue", dil, "%"],
         ["  → percentile (lower dilution better)", pd_, "n"],
         [f"Total assets, FY {fy0}", assets, "$"], [f"Total assets, FY {fy3}", a3, "$"],
         [f"Revenue, FY {fy0}", rev0, "$"], [f"Revenue, FY {fy3}", rev3, "$"],
         ["Bloat = asset CAGR − revenue CAGR (both FY0 vs FY-3)", blo, "%"],
         ["  → percentile (lower bloat better)", pb, "n"]],
        "Hygiene composite", r, "n", m.get("q6_hygiene"),
        "accruals dropped for financials" if fin else "each alone is noisy; together they are the tripwire")

    # ---------------- BIZ MOMENTUM ----------------
    va0, va3 = _f(f.get("w_vol_a0")), _f(f.get("w_vol_a3"))
    cagr = _f(f.get("vol_cagr_3y"))
    y0, y1 = _f(f.get("vol_yoy_q0")), _f(f.get("vol_yoy_q1"))
    vq = f.get("w_vol_q") or []
    dq = f.get("w_date_q") or []
    lab = f.get("w_vol_label", "revenue")
    L = int(f.get("w_lag") or 4)                     # rows back to the same period a year earlier
    per = "half-year" if L == 2 else "quarter"
    r = (y0 - cagr) * 100.0 if _ok(y0, cagr) else NA
    w["b1_vs_trend"] = _entry(f"latest-{per} YoY growth − 3-year CAGR (in points)",
        [[f"{lab.capitalize()}, latest {per} {dq[0] if dq else ''}".strip(), _f(vq[0]) if vq else NA, "$"],
         [f"{lab.capitalize()}, same {per} a year earlier", _f(vq[L]) if len(vq) > L else NA, "$"],
         [f"Latest-{per} YoY growth", y0, "%"],
         [f"{lab.capitalize()}, latest FY {f.get('w_date_a0') or ''}".strip(), va0, "$"],
         [f"{lab.capitalize()}, FY three years earlier {f.get('w_date_a3') or ''}".strip(), va3, "$"],
         ["3-year CAGR = (latest ÷ 3y ago)^(1/3) − 1", cagr, "%"]],
        "Acceleration vs trend", r, "pp", m.get("b1_vs_trend"),
        "acceleration against the company's own base rate")

    need = f.get("b2_min_periods", 6)
    r = (y0 - y1) * 100.0 if _ok(y0, y1) and _f(f.get("volume_quarters")) >= need else NA
    w["b2_sequential"] = _entry(f"latest-{per} YoY − prior-{per} YoY (in points)",
        [[f"{lab.capitalize()}, latest {per}", _f(vq[0]) if len(vq) > 0 else NA, "$"],
         [f"{lab.capitalize()}, same {per} a year earlier", _f(vq[L]) if len(vq) > L else NA, "$"],
         [f"Latest-{per} YoY", y0, "%"],
         [f"{lab.capitalize()}, prior {per}", _f(vq[1]) if len(vq) > 1 else NA, "$"],
         [f"{lab.capitalize()}, prior {per} a year earlier", _f(vq[L + 1]) if len(vq) > L + 1 else NA, "$"],
         [f"Prior-{per} YoY", y1, "%"]],
        "Sequential acceleration", r, "pp", m.get("b2_sequential"),
        f"the second derivative; needs {need} {per}s")

    if fin:
        now, then = _f(f.get("fin_margin_q0")), _f(f.get("fin_margin_q4"))
        r = (now - then) * 10000.0 if _ok(now, then) else NA
        if f.get("fin_margin_basis") == "nim":
            ins = [["Interest income, latest quarter", _f(f.get("w_ii_q0")), "$"],
                   ["Interest expense, latest quarter", _f(f.get("w_ie_q0")), "$"],
                   ["Total assets", _f(f.get("assets")), "$"],
                   ["Spread on assets, latest quarter", now, "%"],
                   ["Interest income, year-earlier quarter", _f(f.get("w_ii_q4")), "$"],
                   ["Interest expense, year-earlier quarter", _f(f.get("w_ie_q4")), "$"],
                   ["Spread on assets, year-earlier quarter", then, "%"]]
            formula = "(interest income − interest expense) ÷ assets, now vs a year earlier (bp)"
            note = "financials variant: a proxy for net interest margin"
        else:
            ins = [["Operating income, latest quarter", _f(f.get("w_oi_q0")), "$"],
                   ["Revenue, latest quarter", _f(f.get("w_rev_q0")), "$"],
                   ["Operating margin, latest quarter", now, "%"],
                   ["Operating income, year-earlier quarter", _f(f.get("w_oi_q4")), "$"],
                   ["Revenue, year-earlier quarter", _f(f.get("w_rev_q4")), "$"],
                   ["Operating margin, year-earlier quarter", then, "%"]]
            formula = "operating margin now − same quarter a year earlier (bp)"
            note = "financials variant: moves opposite to the combined ratio"
        w["b3_margin_delta"] = _entry(formula, ins, "Margin delta", r, "bp",
                                      m.get("b3_margin_delta"), note)
    else:
        g0, r0 = _f(f.get("w_gp_q0")), _f(f.get("w_rev_q0"))
        g4, r4 = _f(f.get("w_gp_q4")), _f(f.get("w_rev_q4"))
        now, then = _f(f.get("gm_q0")), _f(f.get("gm_q4"))
        r = (now - then) * 10000.0 if _ok(now, then) else NA
        w["b3_margin_delta"] = _entry("latest-quarter gross margin − same quarter a year earlier (bp)",
            [["Gross profit, latest quarter", g0, "$"], ["Revenue, latest quarter", r0, "$"],
             ["Gross margin, latest quarter", now, "%"],
             ["Gross profit, year-earlier quarter", g4, "$"], ["Revenue, year-earlier quarter", r4, "$"],
             ["Gross margin, year-earlier quarter", then, "%"]],
            "Gross margin delta", r, "bp", m.get("b3_margin_delta"),
            ("set aside: at least one of these quarters reports no cost of revenue, so its "
             "gross profit equals its revenue. Comparing it with a real quarter would "
             "manufacture a swing that never happened."
             if (_ok(g0) and _ok(r0) and r0 > 0 and g0 / r0 >= C.GP_MAX_MARGIN)
             or (_ok(g4) and _ok(r4) and r4 > 0 and g4 / r4 >= C.GP_MAX_MARGIN)
             else "growth bought with discounts is worth less than growth earned at rising margins"))

    # ---------------- PRICE - STRENGTH ----------------
    for key, win, lab_w in (("p1_trend_12_1", "monthly_returns_12_1", "months −12 to −2"),
                            ("p2_trend_6_1", "monthly_returns_6_1", "months −6 to −2")):
        arr = np.asarray([x for x in (f.get(win) or []) if x is not None and math.isfinite(x)], float)
        mn = float(arr.mean()) if arr.size else NA
        sd = float(arr.std(ddof=1)) if arr.size > 1 else NA
        minobs = C.MOM_MIN_OBS_12_1 if key == "p1_trend_12_1" else C.MOM_MIN_OBS_6_1
        r = mn / sd if arr.size >= minobs and _ok(mn, sd) and sd != 0 else NA
        w[key] = _entry("mean monthly return ÷ standard deviation",
            [[f"Monthly returns, {lab_w}", ", ".join(f"{x*100:+.1f}%" for x in arr), "t"],
             ["Observations", arr.size, "d"], ["Mean monthly return", mn, "%"],
             ["Standard deviation", sd, "%"]],
            "Trend-adjusted momentum", r, "n", m.get(key),
            f"the most recent month is skipped because it mean-reverts; at least {minobs} observations")

    hi = _f(f.get("high_252"))
    r = price / hi - 1 if _ok(price, hi) and hi > 0 else NA
    w["p3_high_distance"] = _entry("price ÷ 252-day high − 1",
        [["Price", price, "$"], ["252-day high", hi, "$"]],
        "Distance from high", r, "%", m.get("p3_high_distance"), "closer to zero is better")

    # ---------------- PRICE - CREDIBILITY ----------------
    sign, up, dn = _f(f.get("period_return_sign")), _f(f.get("pct_up_days")), _f(f.get("pct_down_days"))
    r = sign * (up - dn) if _ok(sign, up, dn) else NA
    w["p4_continuity"] = _entry("sign(period return) × (% up days − % down days)",
        [["Direction of the period return", sign, "n"], ["% of days up", up, "n"],
         ["% of days down", dn, "n"]],
        "Continuity", r, "n", m.get("p4_continuity"),
        "over trading days t−252 to t−21; a move made in many small steps continues")

    daily = [x for x in (f.get("daily_returns_252") or []) if x is not None and math.isfinite(x)]
    top = sorted(daily)[-C.MAX_LOTTERY_DAYS:] if len(daily) >= C.MAX_LOTTERY_DAYS else []
    r = float(np.mean(top)) if top else NA
    w["p5_lottery"] = _entry(f"mean of the {C.MAX_LOTTERY_DAYS} largest daily returns",
        [[f"The {C.MAX_LOTTERY_DAYS} largest daily returns",
          ", ".join(f"{x*100:+.2f}%" for x in reversed(top)), "t"],
         ["Days examined", len(daily), "d"]],
        "Lottery days (MAX5)", r, "%", m.get("p5_lottery"),
        "lower is better; returns concentrated in a few days signal crowding")

    dd = [x for x in (f.get("returns_on_down_days") or []) if x is not None and math.isfinite(x)]
    r = float(np.mean(dd)) if len(dd) >= C.DOWN_DAYS_MIN else NA
    w["p6_down_resilience"] = _entry("mean return on the benchmark's worst 15% of days",
        [["Benchmark down-days examined", len(dd), "d"],
         ["Mean return on those days", float(np.mean(dd)) if dd else NA, "%"]],
        "Down-market resilience", r, "%", m.get("p6_down_resilience"),
        f"over the trailing 252 days; at least {C.DOWN_DAYS_MIN} qualifying days required")

    # set-asides applied after the metrics are computed must be applied here
    # too, or the workings would disagree with the ranked (blank) values
    def _set_aside(keys, why):
        for k in keys:
            if k in w:
                w[k]["out"][1] = None
                w[k]["note"] = why
                w[k]["ok"] = _close(NA, m.get(k))
    if f.get("fx_mismatch"):                              # row 24
        _set_aside(("v1_ebit_ev", "v2_ev_gp", "v3_fwd_earn_yield", "v4_norm_ep",
                    "v5_ev_sales_vs_hist"),
                   "set aside: statements and price are in different currencies and no "
                   "exchange rate was found, so no value ratio can be trusted")
    if f.get("price_stale"):                              # row 34
        _set_aside(("p1_trend_12_1", "p2_trend_6_1", "p3_high_distance", "p4_continuity",
                    "p5_lottery", "p6_down_resilience"),
                   f"set aside: the last price bar is over {C.STALE_PRICE_DAYS} days old")
    return expand(w, f)


def finalize_hygiene(w: dict, pct_acc, pct_dil, pct_blo, q6) -> dict:
    """Hygiene is a mean of three percentiles, and percentiles only exist once
    the whole universe is ranked - so this entry is completed in the scoring
    step, after build() has run for every name."""
    e = w.get("q6_hygiene")
    if not e:
        return w
    vals = {"  → percentile (lower accruals better)": pct_acc,
            "  → percentile (lower dilution better)": pct_dil,
            "  → percentile (lower bloat better)": pct_blo}
    for row in e["in"]:
        if row[0] in vals:
            v = _f(vals[row[0]])
            row[1] = None if not _ok(v) else float(v)
    parts = [_f(x) for x in (pct_acc, pct_dil, pct_blo) if _ok(_f(x))]
    r = float(np.mean(parts)) if parts else NA
    e["out"][1] = None if not _ok(r) else r
    e["ok"] = _close(r, q6)
    return w


# ---------------------------------------------------------------------------
# bottom-up expansion
# ---------------------------------------------------------------------------
def _money(v):
    v = _f(v)
    if not _ok(v):
        return "—"
    a, sign = abs(v), "−" if v < 0 else ""
    if a >= 1e12:
        return f"{sign}${a/1e12:.2f}T"
    if a >= 1e9:
        return f"{sign}${a/1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a/1e6:.1f}M"
    return f"{sign}${a:,.0f}"


# every TTM line, mapped to the statement field whose four quarters make it up
TTM_FIELDS = {
    "Operating income, TTM": "operatingIncome",
    "Gross profit, TTM": "grossProfit",
    "Revenue, TTM": "revenue",
    "Net income, TTM": "netIncome",
    "Operating cash flow, TTM": "operatingCashFlow",
    "Stock comp, TTM": "stockBasedCompensation",
}


def expand(w: dict, f: dict) -> dict:
    """Unfold every aggregate into the vendor rows it came from.

    A TTM figure becomes its four quarters; a 5-year average becomes each year;
    an own-history median becomes each annual point. The four quarters are also
    re-summed and checked against the TTM figure the model used - a quarter
    missing or double-counted would otherwise pass unnoticed.
    """
    quarters = f.get("w_q") or {}
    for key, e in w.items():
        new_in = []
        for row in e["in"]:
            label = row[0]
            field = TTM_FIELDS.get(label)
            if field and quarters.get(field):
                qs = quarters[field]
                unit = "half-year" if f.get("cadence") == "half-yearly" else "quarter"
                kids = [[f"{unit} ending {d or '—'}", _conv(v, "$"), "$"] for d, v in qs]
                total = sum(_f(v) for _, v in qs if _ok(_f(v)))
                if (len(qs) == int(f.get("ttm_n") or 4) and all(_ok(_f(v)) for _, v in qs)
                        and row[1] is not None):
                    if not _close(total, row[1]):
                        e["ok"] = False          # the quarters do not add up to the TTM used
                row = row[:3] + [kids]
            elif label.startswith("Annual net margins") and f.get("w_margin_years"):
                kids = [[f"FY {d} · net income {_money(n)} ÷ revenue {_money(r)}", m, "%"]
                        for d, n, r, m in f["w_margin_years"]]
                row = ["Net margin by year", None, "t", kids]
            elif label.startswith("Annual ROE") and f.get("w_roe_years"):
                kids = [[f"FY {d} · net income {_money(n)} ÷ equity {_money(q)}", r, "%"]
                        for d, n, q, r in f["w_roe_years"]]
                row = ["ROE by year", None, "t", kids]
            elif label == "Consensus NTM EPS" and f.get("w_ntm_parts") and f.get("fwd_basis") == "ntm":
                kids = [[f"FY ending {d} · consensus EPS {v:.3f} × weight {wt:.0%}", v * wt, "n"]
                        for d, v, wt in f["w_ntm_parts"]]
                parts_sum = sum(v * wt for _, v, wt in f["w_ntm_parts"])
                if row[1] is not None and not _close(parts_sum, row[1]):
                    e["ok"] = False              # the blend does not reproduce the EPS used
                row = row[:3] + [kids]
            elif label.startswith("Own 5-year median") and f.get("w_hist_points"):
                pts = f["w_hist_points"]
                if f.get("is_fin"):
                    kids = [[f"FY {d} · mcap {_money(mc)} ÷ book {_money(eq)}", r, "x"]
                            for d, mc, eq, _, _, r in pts]
                else:
                    kids = [[f"FY {d} · (mcap {_money(mc)} + debt {_money(dd)} − cash "
                             f"{_money(cc)}) ÷ revenue {_money(rv)}", r, "x"]
                            for d, mc, dd, cc, rv, r in pts]
                row = row[:3] + [kids]
            new_in.append(row)
        e["in"] = new_in
    return w
