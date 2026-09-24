"""
VQMB — the metric library (handbook Part II, §12).

Twenty metrics. Each returns a raw value or NaN, and every NaN is deliberate:
either blank-by-inapplicability (silent, the rule working) or blank-by-youth
(raises SHORT HISTORY). Never a silent zero — a zero ranks, a blank reweights.

Conventions
    TTM   trailing twelve months, assembled from the four most recent quarters
    mcap  market capitalisation
    EV    mcap + total debt − cash & equivalents
    inv   lower raw ranks higher (applied later, in the percentile step)

Financials variant (§12) applies only where the balance sheet IS the business:
banks, NBFCs/lenders, insurers. Exchanges, asset managers, brokers and payment
processors stay on the standard set — so Visa, Mastercard, S&P Global, BlackRock
and the exchanges are scored as ordinary companies.

PPOP / NIM / GWP / combined ratio are not in FMP's standardised statements.
Agreed substitutes, verified against real S&P financials:
    PPOP     -> revenue − interestExpense − operatingExpenses
    GWP      -> revenue (premiums plus investment income)
    NIM      -> (interestIncome − interestExpense) / average total assets
    combined -> operating margin, direction inverted
Provisioning quality stays invisible; the analyst note is mandatory.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C

NA = float("nan")


def _f(x):
    """Float or NaN. Treats None, '', and non-numeric junk as missing."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NA
    return v if math.isfinite(v) else NA


def _ok(*vals):
    return all(math.isfinite(v) for v in vals)


def _safe_div(a, b, *, require_positive_denom=True):
    a, b = _f(a), _f(b)
    if not _ok(a, b) or b == 0:
        return NA
    if require_positive_denom and b < 0:
        return NA
    return a / b


def _has_at_least(value, n) -> bool:
    """Row 19: present AND at least n. A missing count must mean 'not enough';
    a bare `x < n` test is False for NaN and silently lets a missing count pass."""
    v = _f(value)
    return _ok(v) and v >= n


def _cagr(now, then, years):
    now, then = _f(now), _f(then)
    if not _ok(now, then) or then <= 0 or now <= 0 or years <= 0:
        return NA
    return (now / then) ** (1.0 / years) - 1.0


# ===========================================================================
# VALUE — five lenses, equal weight (§5)
# ===========================================================================
def v1_ebit_ev(f: dict) -> float:
    """EBIT / EV, higher better. The acquirer's yield: what the whole
    enterprise earns on what it costs to own it, capital-structure neutral.

    EV <= 0 (net cash exceeds market cap) -> blank. Rare, and V4 still catches
    the cheapness. Financials: E/P instead, because a lender has no meaningful
    enterprise value — its debt is raw material, not financing."""
    if f.get("is_fin"):
        return _safe_div(f.get("ni_ttm"), f.get("mcap"))
    ev = _f(f.get("ev"))
    if not _ok(ev) or ev <= 0:
        return NA
    return _safe_div(f.get("ebit_ttm"), ev)


def v2_ev_gp(f: dict) -> float:
    """EV / gross profit, inv. The Meta-catcher: a company expensing heavy
    growth investment through the P&L looks dear on EBIT and cheap on gross
    profit, which is the least-manipulated profitability line. Finds good
    businesses whose earnings are suppressed by their own spending.

    Financials: book / price instead (direction flips to higher-better, see
    FIN_DIRECTION_OVERRIDE). GP <= 0 -> blank."""
    if f.get("is_fin"):
        book = _f(f.get("book_tangible") if f.get("fin_kind") == "insurer"
                  else f.get("book"))
        return _safe_div(book, f.get("mcap"))
    gp, ev = _f(f.get("gp_ttm")), _f(f.get("ev"))
    if not _ok(gp) or gp <= 0:
        return NA
    # row 2: a negative EV gives a negative EV/GP that would rank as the
    # cheapest name in the universe - blank it, the same rule as V1
    if not _ok(ev) or ev <= 0:
        return NA
    return ev / gp


def v3_fwd_earn_yield(f: dict) -> tuple[float, dict]:
    """NTM EPS / price, higher better, via the three-path cascade.

    CONSENSUS — fiscal-year estimates blended into a true next-twelve-months
      figure. Without the blend the horizon depends on where a company sits in
      its own accounting calendar: a September year-end gets scored on a year
      almost over, a June year-end on one ten months out. That is a calendar
      artefact, not a valuation signal.
    CALC — no consensus: revenue x (1+g) x 5y average net margin / shares.
      Guard: 5y average margin <= 0 -> blank, tag CALC-NA. Never emit a
      negative modelled yield.
    LOSS — consensus forecasts a loss: blank, tag LOSS, reweight. Never
      substitute history for a forecast loss; that would hand a loss-making
      company the profitability it used to have.

    Growth belongs inside value (commitment 2), so this is where forward growth
    enters the model.
    """
    tags = {"v3_calc": False, "v3_loss": False, "v3_calc_na": False,
            "fwd_basis": "none", "ntm_coverage": NA}

    eps = _f(f.get("ntm_eps"))
    cov = _f(f.get("ntm_coverage"))
    price = _f(f.get("price"))
    tags["ntm_coverage"] = cov
    if _ok(eps):
        tags["fwd_basis"] = f.get("fwd_basis") or "ntm"
        if eps <= 0:
            tags["v3_loss"] = True
            return NA, tags
        y = _safe_div(eps, price)
        if _ok(y):
            return y, tags

    # CALC path
    tags["v3_calc"] = True
    tags["fwd_basis"] = "calc"
    g = _f(f.get("g_projection"))
    if not _ok(g):
        # row 14: a missing growth figure is missing, not zero growth
        tags["v3_calc_na"] = True
        return NA, tags
    if f.get("is_fin"):
        roe5, book = _f(f.get("roe_5y_avg")), _f(f.get("book"))
        if not _ok(roe5) or roe5 <= 0 or not _has_at_least(f.get("roe_years"), C.MIN_AVG_YEARS):
            tags["v3_calc_na"] = True
            return NA, tags
        return _safe_div(roe5 * book * (1 + g), f.get("mcap")), tags
    margin5 = _f(f.get("net_margin_5y_avg"))
    if (not _ok(margin5) or margin5 <= 0
            or not _has_at_least(f.get("margin_years"), C.MIN_AVG_YEARS)):
        tags["v3_calc_na"] = True
        return NA, tags
    return _safe_div(margin5 * _f(f.get("rev_ttm")) * (1 + g), f.get("mcap")), tags


def v4_norm_ep(f: dict) -> float:
    """(5y average net margin x TTM revenue) / mcap, higher better.

    The cycle lens — mid-cycle margins on current revenue answers "what does
    this earn through a cycle?", the anti-peak-margin device. When the spot
    lenses (V1, V3) and the cycle lenses (V4, V5) disagree widely, the value
    pillar is making a margin bet and the card must say so.

    Fewer than 3 years of margin history -> blank."""
    if f.get("is_fin"):
        roe5, book = _f(f.get("roe_5y_avg")), _f(f.get("book"))
        if not _ok(roe5, book) or not _has_at_least(f.get("roe_years"), C.MIN_AVG_YEARS):
            return NA
        return _safe_div(roe5 * book, f.get("mcap"))
    if not _has_at_least(f.get("margin_years"), C.MIN_AVG_YEARS):
        return NA
    m5 = _f(f.get("net_margin_5y_avg"))
    if not _ok(m5):
        return NA
    return _safe_div(m5 * _f(f.get("rev_ttm")), f.get("mcap"))


def v5_ev_sales_vs_hist(f: dict) -> float:
    """(current EV/sales / own 5y median) - 1, inv. The only self-referential
    lens: cheap against its own past, which catches deratings and froth that
    cross-sectional lenses cannot see. Financials use P/B against their own
    median. Too little listed history -> blank and SHORT HISTORY."""
    if f.get("is_fin"):
        cur = _safe_div(f.get("mcap"), f.get("book"))
        med = _f(f.get("pb_median_hist"))
    else:
        ev = _f(f.get("ev"))
        if not _ok(ev) or ev <= 0:          # row 9: a negative EV reads as deeply cheap
            return NA
        cur = _safe_div(ev, f.get("rev_ttm"))
        med = _f(f.get("ev_sales_median_hist"))
    if not _ok(cur, med) or med <= 0:
        return NA
    return cur / med - 1.0


# ===========================================================================
# QUALITY — engine x shield (§6)
# ===========================================================================
def effective_tax(rate) -> float:
    """A company's effective rate, clamped to 0-35% (§6). Falls back to the
    statutory-ish 21% only when the rate cannot be computed at all."""
    r = _f(rate)
    if not _ok(r):
        return C.NOPAT_TAX_RATE
    return min(max(r, C.TAX_CLAMP[0]), C.TAX_CLAMP[1])


def q1_roic(f: dict) -> tuple[float, bool]:
    """NOPAT / invested capital. What the business earns on the capital inside
    it — the compounding rate of the machine.

    NOPAT = EBIT x (1 - effective tax, clamped 0-35%); IC = equity + net debt.
    The company's OWN rate, per handbook §6 - a persistently low-tax business
    genuinely keeps more of what it earns, and the clamp is what stops a
    one-off credit or charge from running away with the number.
    EBIT <= 0 or IC <= 0 -> fall back to gross profit / assets in the same
    column, marked, so young or loss-making names still get a productivity
    read instead of a hole. Financials: ROE."""
    if f.get("is_fin"):
        return _safe_div(f.get("ni_ttm"), f.get("book")), False
    ebit, ic = _f(f.get("ebit_ttm")), _f(f.get("invested_capital"))
    if _ok(ebit, ic) and ebit > 0 and ic > 0:
        return (ebit * (1 - effective_tax(f.get("effective_tax_rate")))) / ic, False

    if C.ROIC_LOSS_MODE == "fallback":                # the pre-register behaviour
        return _safe_div(f.get("gp_ttm"), f.get("assets")), True

    # Register row 1. GP/assets (~0.4) sits on a larger scale than ROIC (~0.15),
    # so switching a loss-maker onto it ranked loss-makers near the top of Q1.
    if not _ok(ebit, ic):
        return NA, False                              # missing EBIT or IC: blank
    if ebit <= 0 and ic > 0:
        return ebit / ic, False                       # a real negative return, no tax credit
    if ebit > 0 and ic <= 0:
        # ROIC is genuinely undefined here (negative capital), so a productivity
        # read stands in - marked, and flagged GP/A on the screen
        return _safe_div(f.get("gp_ttm"), f.get("assets")), True
    return NA, False                                  # EBIT <= 0 and IC <= 0: blank


def q2_roiic(f: dict) -> float:
    """(NOPAT_t - NOPAT_t-3y) / (IC_t - IC_t-3y). The forward-looking half of
    quality: what NEW capital earns, and the best single tell of whether growth
    creates or destroys value.

    Change in invested capital below 5% of IC, or negative (buybacks shrinking
    the capital base), -> blank by inapplicability. The rule working, not
    missing data: there is no incremental return on capital that did not move.
    Financials: incremental ROE = change in net income / change in book."""
    if f.get("is_fin"):
        d_ni, d_book = _f(f.get("d_ni_3y")), _f(f.get("d_book_3y"))
        if not _ok(d_ni, d_book) or d_book <= 0:
            return NA
        return d_ni / d_book
    d_nopat, d_ic, ic = _f(f.get("d_nopat_3y")), _f(f.get("d_ic_3y")), _f(f.get("invested_capital"))
    if not _ok(d_nopat, d_ic, ic) or ic <= 0:
        return NA
    if d_ic <= 0 or d_ic < C.ROIIC_MIN_DELTA_IC * ic:
        return NA
    return d_nopat / d_ic


def q3_persistence(f: dict) -> float:
    """Share of the last 8 quarters with positive year-on-year growth in the
    volume line. Consistency is a quality trait; also reused as the CALC uncap
    test in V3. Fewer than 8 quarters: compute on at least 4, else blank and
    SHORT HISTORY."""
    hits, total = _f(f.get("persistence_hits")), _f(f.get("persistence_quarters"))
    need = f.get("persistence_min", C.PERSISTENCE_MIN_QUARTERS)   # halves need half as many
    if not _ok(hits) or not _has_at_least(total, need) or total <= 0:
        return NA
    return hits / total


def q4_leverage_score(f: dict) -> tuple[float, bool]:
    """Net debt / EBIT, scored 0-100, plus the gate.

    Net cash -> 100. 0 to 3x falls linearly 100 -> 60. Above 3x it turns convex,
    60 x (1 - (L-3)^2), reaching 0 at 4x, where the gate fires. The kink is the
    point: leverage is benign until it isn't, and the danger zone is narrow and
    non-linear, so a linear score would understate it badly.

    EBIT <= 0 with net debt -> gate. EBIT <= 0 with net cash -> blank (no
    earnings to measure against, but no danger either).

    Financials: equity / assets scored instead, with its own capital gate."""
    if f.get("is_fin"):
        cap = _f(f.get("capital_ratio"))
        if not _ok(cap):
            return NA, False
        floor = (C.INSURER_CAPITAL_GATE if f.get("fin_kind") == "insurer"
                 else C.BANK_CAPITAL_GATE)
        if cap < floor:
            return 0.0, True
        # 100 at 3x the floor or better, scaling down to 0 at the floor itself
        return float(np.clip((cap - floor) / (2 * floor) * 100.0, 0, 100)), False

    nd, ebit = _f(f.get("net_debt")), _f(f.get("ebit_ttm"))
    # The EBIT test comes FIRST. A loss-maker sitting on net cash is not a
    # fortress - there are no earnings to service anything with - so the
    # handbook blanks it rather than awarding 100. Checking net cash first
    # would hand every cash-rich loss-maker a perfect leverage score.
    if not _ok(ebit):
        # row 11: a MISSING operating profit is missing data, not a loss
        return NA, bool(C.MISSING_EBIT_GATES and _ok(nd) and nd > 0)
    if ebit <= 0:
        if _ok(nd) and nd > 0:
            return NA, True                       # reported loss with net debt -> gate
        return NA, False                          # net cash and no earnings -> blank
    if _ok(nd) and nd <= 0:
        return 100.0, False                       # net cash, earnings positive
    L = nd / ebit
    if L > C.LEV_MAX:
        return 0.0, True
    if L <= C.LEV_KINK:
        return float(100.0 - (100.0 - C.LEV_SCORE_AT_KINK) * (L / C.LEV_KINK)), False
    return float(C.LEV_SCORE_AT_KINK * (1 - (L - C.LEV_KINK) ** 2)), False


def q5_drawdown(f: dict) -> float:
    """Episode count + |worst| / 50, inv. The market's own memory of fragility:
    how often and how hard this name breaks. An episode opens at -30% from a
    peak and closes on recovery to within 10% of it. Computed on USD prices so
    an emerging-market name cannot hide behind its currency.

    Too little history -> blank and SHORT HISTORY: 'no winters on record' is a
    mandatory analyst mention, because absence of scars is not the same as
    resilience."""
    n, worst, years = (_f(f.get("dd_episodes")), _f(f.get("dd_worst")),
                       _f(f.get("price_years")))
    if not _ok(n, years) or years < C.DRAWDOWN_MIN_YEARS:
        return NA
    if not _ok(worst):
        worst = 0.0
    return n + abs(worst) * 100.0 / 50.0


def q6_hygiene(f: dict, pcts: dict | None = None) -> float:
    """Mean of three inv sub-percentiles: accruals, dilution, bloat.

    accruals  (TTM net income - TTM operating cash flow) / assets — earnings
              not backed by cash
    dilution  3y diluted share CAGR + TTM SBC / revenue — who the growth is for
    bloat     3y asset CAGR - 3y revenue CAGR — the balance sheet outgrowing
              the business

    One shared weight because each alone is noisy; together they are the
    accounting-quality tripwire behind the HYGIENE flag. Financials drop
    accruals (no provisioning data) and use loan-vs-deposit growth for bloat;
    here that substitutes to assets-vs-revenue.

    Called twice: once to emit the three raw components, then again with their
    universe percentiles to average them."""
    if pcts is None:
        return NA
    vals = [v for v in (pcts.get("accruals"), pcts.get("dilution"),
                        pcts.get("bloat")) if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else NA


def hygiene_components(f: dict) -> dict:
    """The three raw inputs to Q6, each 'higher is worse'."""
    out = {}
    if f.get("is_fin"):
        out["accruals"] = NA          # dropped: no provisioning data
    else:
        out["accruals"] = _safe_div(
            _f(f.get("ni_ttm")) - _f(f.get("cfo_ttm")), f.get("assets"))
    # row 29: share CAGR on ANNUAL diluted shares, FY0 against FY-3 - not the
    # latest quarter against an annual figure
    share_cagr = _cagr(f.get("shares_a0"), f.get("shares_3y"), 3)
    sbc = _safe_div(f.get("sbc_ttm"), f.get("rev_ttm"))
    if C.DILUTION_NEEDS_BOTH:
        # row 15: a missing half is missing, not zero dilution
        out["dilution"] = share_cagr + sbc if _ok(share_cagr, sbc) else NA
    elif _ok(share_cagr) or _ok(sbc):
        out["dilution"] = (0.0 if not _ok(share_cagr) else share_cagr) + \
                          (0.0 if not _ok(sbc) else sbc)
    else:
        out["dilution"] = NA
    # row 29: both CAGRs from the annual statements, FY0 against FY-3
    a = _cagr(f.get("assets"), f.get("assets_3y"), 3)
    r = _cagr(f.get("rev_a0"), f.get("rev_3y"), 3)
    out["bloat"] = a - r if _ok(a, r) else NA
    return out


# ===========================================================================
# BIZ MOMENTUM — accelerations, equal weight (§7)
# ===========================================================================
# Volume line: quarterly revenue. Banks use PPOP, insurers GWP - substituted
# per the note at the top of this file. B4 (revisions) is removed; without
# daily consensus snapshots its inputs do not exist.
def b1_vs_trend(f: dict) -> float:
    """Latest-quarter volume YoY (pp) minus the 3-year volume CAGR (pp).
    Acceleration against the company's own base rate, not the market's."""
    yoy, cagr3 = _f(f.get("vol_yoy_q0")), _f(f.get("vol_cagr_3y"))
    if not _ok(yoy, cagr3):
        return NA
    return (yoy - cagr3) * 100.0


def b2_sequential(f: dict) -> float:
    """Latest-quarter YoY minus prior-quarter YoY (pp). The second derivative —
    turning points show here first. Needs 6 quarters of history."""
    q0, q1 = _f(f.get("vol_yoy_q0")), _f(f.get("vol_yoy_q1"))
    if not _ok(q0, q1) or not _has_at_least(f.get("volume_quarters"), f.get("b2_min_periods", 6)):
        return NA
    return (q0 - q1) * 100.0


def b3_margin_delta(f: dict) -> float:
    """Latest-quarter gross margin minus the same quarter a year ago (bp).
    Growth bought with discounts — revenue accelerating while margin falls — is
    worth less than growth earned at rising margins.

    Financials: change in operating margin, which for a bank captures spread
    compression and cost discipline together, and for an insurer moves opposite
    to the combined ratio."""
    now, then = (_f(f.get("fin_margin_q0")), _f(f.get("fin_margin_q4"))) if f.get("is_fin") \
        else (_f(f.get("gm_q0")), _f(f.get("gm_q4")))
    if not _ok(now, then):
        return NA
    return (now - then) * 10000.0


# ===========================================================================
# PRICE MOMENTUM — strength x credibility (§8)
# ===========================================================================
def p1_trend_12_1(f: dict) -> float:
    """Mean of monthly returns m-12..m-2 divided by their standard deviation.
    The canonical momentum window, volatility-honest — a steady climb beats the
    same total return delivered in lurches. Skip-month convention: the most
    recent month is excluded because it mean-reverts."""
    return _trend(f.get("monthly_returns_12_1"), C.MOM_MIN_OBS_12_1)


def p2_trend_6_1(f: dict) -> float:
    """The same over m-6..m-2 — the faster clock. With P1 it gives an old trend
    and a young one."""
    return _trend(f.get("monthly_returns_6_1"), C.MOM_MIN_OBS_6_1)


def _trend(returns, min_obs) -> float:
    if returns is None:
        return NA
    r = np.asarray([x for x in returns if x is not None and math.isfinite(x)], dtype=float)
    if r.size < min_obs:
        return NA
    sd = r.std(ddof=1)
    if not math.isfinite(sd) or sd == 0:
        return NA
    return float(r.mean() / sd)


def p3_high_distance(f: dict) -> float:
    """price / 252-day max - 1, closer to zero better. The anchoring effect,
    and the Darvas eligibility input."""
    ratio = _safe_div(f.get("price"), f.get("high_252"))
    return ratio - 1.0 if _ok(ratio) else NA


def p4_continuity(f: dict) -> float:
    """sign(period return) x (% up days - % down days) over t-252..t-21.

    Frog-in-the-pan: a move delivered in many small steps continues, the same
    move delivered in a few jumps does not — information discreteness.

    NOTE the handbook flags this formulation as cited from memory. It is
    implemented exactly as written, per instruction. If the credibility
    dampener behaves oddly, the sign convention here is the first thing to
    check against Da-Gurun-Warachka."""
    sign, up, down = (_f(f.get("period_return_sign")), _f(f.get("pct_up_days")),
                      _f(f.get("pct_down_days")))
    if not _ok(sign, up, down):
        return NA
    return sign * (up - down)


def p5_lottery(f: dict) -> float:
    """Mean of the 5 largest daily returns over the trailing 252 days, inv.
    Returns concentrated in lottery days signal crowding and reversal risk.

    NOTE also flagged in the handbook as cited from memory — the original
    measures this over one month, not a year. Implemented as written, per
    instruction."""
    r = f.get("daily_returns_252")
    if r is None:
        return NA
    arr = np.asarray([x for x in r if x is not None and math.isfinite(x)], dtype=float)
    if arr.size < C.MAX_LOTTERY_DAYS:
        return NA
    return float(np.sort(arr)[-C.MAX_LOTTERY_DAYS:].mean())


def p6_down_resilience(f: dict) -> float:
    """Mean stock return on the benchmark's worst-15% days over the trailing
    252 days. Who holds it when it matters — the institutional-hands read.
    Fewer than 8 qualifying days -> blank."""
    rets = f.get("returns_on_down_days")
    if rets is None:
        return NA
    arr = np.asarray([x for x in rets if x is not None and math.isfinite(x)], dtype=float)
    if arr.size < C.DOWN_DAYS_MIN:
        return NA
    return float(arr.mean())


# ===========================================================================
# assembly
# ===========================================================================
def compute_all(f: dict) -> dict:
    """Every metric for one name, plus the tags and gate signals the chain
    needs downstream. `f` is the normalised fact sheet built by the data layer.
    """
    out: dict = {"symbol": f.get("symbol")}
    out["is_fin"] = bool(f.get("is_fin"))
    out["fin_kind"] = f.get("fin_kind")

    out["v1_ebit_ev"] = v1_ebit_ev(f)
    out["v2_ev_gp"] = v2_ev_gp(f)
    v3, tags = v3_fwd_earn_yield(f)
    out["v3_fwd_earn_yield"] = v3
    out.update(tags)
    out["v4_norm_ep"] = v4_norm_ep(f)
    out["v5_ev_sales_vs_hist"] = v5_ev_sales_vs_hist(f)

    roic, fallback = q1_roic(f)
    out["q1_roic"] = roic
    out["q1_used_gpa_fallback"] = fallback
    out["q2_roiic"] = q2_roiic(f)
    out["q3_persistence"] = q3_persistence(f)

    lev, gate = q4_leverage_score(f)
    out["q4_leverage_score"] = lev
    out["leverage_gate"] = gate
    out["q5_drawdown"] = q5_drawdown(f)
    out.update({f"hyg_{k}": v for k, v in hygiene_components(f).items()})

    out["b1_vs_trend"] = b1_vs_trend(f)
    out["b2_sequential"] = b2_sequential(f)
    out["b3_margin_delta"] = b3_margin_delta(f)

    out["p1_trend_12_1"] = p1_trend_12_1(f)
    out["p2_trend_6_1"] = p2_trend_6_1(f)
    out["p3_high_distance"] = p3_high_distance(f)
    out["p4_continuity"] = p4_continuity(f)
    out["p5_lottery"] = p5_lottery(f)
    out["p6_down_resilience"] = p6_down_resilience(f)

    # row 24: statements and price in different currencies with no rate - every
    # value lens would divide one currency by another, so all five are set aside
    out["fx_mismatch"] = bool(f.get("fx_mismatch"))
    if out["fx_mismatch"]:
        for k in ("v1_ebit_ev", "v2_ev_gp", "v3_fwd_earn_yield", "v4_norm_ep",
                  "v5_ev_sales_vs_hist"):
            out[k] = NA
    # row 34: a stale last price bar blanks every price metric
    out["price_stale"] = bool(f.get("price_stale"))
    if out["price_stale"]:
        for k in ("p1_trend_12_1", "p2_trend_6_1", "p3_high_distance", "p4_continuity",
                  "p5_lottery", "p6_down_resilience"):
            out[k] = NA
    # data-quality provenance, read by the flags and run.json
    out["stale"] = bool(f.get("stale_statement") or f.get("gapped") or f.get("price_stale"))
    out["stale_reason"] = ", ".join(r for r, on in (
        ("statement over %d days old" % C.STALE_STATEMENT_DAYS, f.get("stale_statement")),
        ("reports not consecutive", f.get("gapped")),
        ("last price over %d days old" % C.STALE_PRICE_DAYS, f.get("price_stale"))) if on)
    out["px_mismatch"] = bool(f.get("px_mismatch"))
    out["cadence"] = f.get("cadence")
    out["implausible_moves"] = f.get("implausible_moves") or []

    # gate inputs and provenance the chain reads later
    out["nd_ebit"] = _safe_div(f.get("net_debt"), f.get("ebit_ttm"),
                               require_positive_denom=True)
    out["ebit_negative_with_net_debt"] = bool(
        _ok(_f(f.get("ebit_ttm"))) and _f(f.get("ebit_ttm")) <= 0
        and _ok(_f(f.get("net_debt"))) and _f(f.get("net_debt")) > 0)
    out["capital_ratio"] = _f(f.get("capital_ratio"))
    # row 19: a MISSING count means not enough history, so each test is
    # "present and at least N" rather than a bare comparison NaN would pass
    out["short_history"] = bool(
        not _has_at_least(f.get("price_years"), C.DRAWDOWN_MIN_YEARS)
        or not _has_at_least(f.get("persistence_quarters"), f.get("persistence_window", 8))
        or not _ok(_f(f.get("ev_sales_median_hist")) if not f.get("is_fin")
                   else _f(f.get("pb_median_hist"))))
    return out
