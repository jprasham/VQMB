"""
VQMB — the fact sheet.

Turns FMP responses into the flat "fact sheet" that metrics.compute_all()
consumes. Every derived quantity the metrics need is assembled here, so the
metric functions stay pure arithmetic with no knowledge of FMP's shapes.

Two things this layer owns that are easy to get wrong:

  The volume line. Handbook §7 wants quarterly revenue for industrials, PPOP
  for banks, GWP for insurers. FMP's standardised statements carry none of the
  bank lines, so the agreed substitutes live here:
      PPOP  -> revenue - interestExpense - operatingExpenses
      GWP   -> revenue
  Verified against real filings: JPM $86.8bn vs ~$86bn reported, WFC $28.9bn vs
  ~$27.7bn. BAC reads high. B1/B2 measure the CHANGE in this line, so a
  consistent overstatement cancels; the level is not used anywhere.

  Own-history medians (V5). Rather than reconstructing daily enterprise value
  over five years, EV/sales is computed at each of the last five annual
  statement dates, using the market cap on that date and that year's balance
  sheet. Five clean points, one median - far less machinery than a daily series
  and not meaningfully less accurate for a median.

Period-end dates are used throughout, not filing dates. Fine for live
screening; any backtest built on this is void by the handbook's own rule (§17).
"""
from __future__ import annotations

import datetime as dt
import math
import statistics

import numpy as np

from . import config as C
from .metrics import effective_tax as M_effective_tax

NA = float("nan")
# The as-of date for staleness checks and the NTM blend. Reset to today at the
# start of every engine.run(), so a long-lived process never scores against the
# date it was imported on.
TODAY = dt.date.today()


# ---------------------------------------------------------------------------
# financials classification (§4, §12)
# ---------------------------------------------------------------------------
# The variant applies ONLY where the balance sheet is the business. Exchanges,
# asset managers, brokers and payment processors stay on the standard set - so
# Visa, Mastercard, S&P Global, BlackRock, CME and ICE are scored as ordinary
# companies, which is the handbook's explicit instruction.
BANK_WORDS = ("bank", "credit services", "mortgage", "consumer finance",
              "savings", "thrift", "lending", "specialty finance")
INSURER_WORDS = ("insurance", "insurer", "reinsur")
NEVER_FIN = ("asset management", "capital markets", "financial data",
             "stock exchange", "exchanges", "brokerage", "broker",
             "investment banking", "shell companies", "conglomerates",
             "payment", "financial conglomerates")


# "Credit Services" is the same FMP industry string for Capital One and for
# Visa. One funds a loan book; the other runs a network and lends nothing. The
# industry text cannot separate them, so a balance-sheet test does: a lender
# pays to fund itself, and its interest expense is a large share of revenue
# (JPM 35%, BAC 41%, WFC 32%), while a payment network's is negligible (~1%).
LENDING_INTEREST_SHARE = 0.10


def classify_financial(sector: str | None, industry: str | None,
                       interest_share: float | None = None) -> tuple[bool, str | None]:
    """Is the balance sheet the business? Only then does the variant apply."""
    ind = (industry or "").lower()
    sec = (sector or "").lower()
    if any(w in ind for w in NEVER_FIN):
        return False, None
    if any(w in ind for w in INSURER_WORDS):
        return True, "insurer"
    if any(w in ind for w in BANK_WORDS):
        if "bank" in ind or "savings" in ind or "thrift" in ind:
            return True, "bank"
        # ambiguous string: require evidence of a funded loan book
        if interest_share is None or not math.isfinite(interest_share):
            return False, None          # no evidence -> standard set, not a guess
        return (True, "nbfc") if interest_share >= LENDING_INTEREST_SHARE else (False, None)
    # sector says financial but the industry string is unrecognised: leave it on
    # the standard set rather than guess, and let the FIN flag stay off
    if sec.startswith("financial"):
        return False, None
    return False, None


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NA
    return v if math.isfinite(v) else NA


def _ok(*v):
    return all(math.isfinite(x) for x in v)


def _rows(blob, key):
    r = blob.get(key)
    return r if isinstance(r, list) else []


def _sum(rows, field, n=4, offset=0):
    vals = [_f(r.get(field)) for r in rows[offset:offset + n]]
    if len(vals) < n or not all(math.isfinite(v) for v in vals):
        return NA
    return float(sum(vals))


def _get(rows, i, field):
    try:
        return _f(rows[i].get(field))
    except (IndexError, AttributeError):
        return NA


def _date(rows, i):
    try:
        return str(rows[i].get("date"))[:10]
    except (IndexError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# reporting cadence and contiguity (register row 8)
# ---------------------------------------------------------------------------
def _pdate(v):
    try:
        return dt.date.fromisoformat(str(v)[:10])
    except (TypeError, ValueError):
        return None


def normalise_periods(rows: list) -> list:
    """Newest first, one row per report date. Vendor order is not trusted: a
    TTM built from rows supplied out of order sums the wrong quarters."""
    seen, out = set(), []
    for r in rows:
        d = _pdate(r.get("date")) if isinstance(r, dict) else None
        if d is None or d in seen:
            continue
        seen.add(d)
        out.append((d, r))
    out.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in out]


def detect_cadence(rows: list) -> dict:
    """Quarterly or half-yearly, from the median gap between report dates.
    n = periods in a TTM window; lag = periods back to the same period a year
    earlier. A half-yearly reporter's TTM is its last two halves, not four."""
    ds = [_pdate(r.get("date")) for r in rows[:9]]
    gaps = [(ds[i] - ds[i + 1]).days for i in range(len(ds) - 1) if ds[i] and ds[i + 1]]
    q, h = C.CADENCE_QUARTERLY_DAYS, C.CADENCE_HALFYEAR_DAYS
    if gaps:
        med = statistics.median(gaps)
        if h[0] <= med <= h[1]:
            return {"kind": "half-yearly", "n": 2, "lag": 2, "band": h}
        if q[0] <= med <= q[1]:
            return {"kind": "quarterly", "n": 4, "lag": 4, "band": q}
        return {"kind": "irregular", "n": 4, "lag": 4, "band": q}
    return {"kind": "unknown", "n": 4, "lag": 4, "band": q}


def window_ok(rows: list, offset: int, n: int, band: tuple) -> bool:
    """n consecutive reports starting at offset, every gap inside the cadence
    band. A missing report inside the window fails it."""
    if len(rows) < offset + n:
        return False
    ds = [_pdate(rows[offset + i].get("date")) for i in range(n)]
    if any(d is None for d in ds):
        return False
    return all(band[0] <= (ds[i] - ds[i + 1]).days <= band[1] for i in range(n - 1))


def pair_ok(rows: list, i: int, j: int) -> bool:
    """A year-on-year pair must actually be about a year apart."""
    if len(rows) <= max(i, j):
        return False
    a, b = _pdate(rows[i].get("date")), _pdate(rows[j].get("date"))
    if not a or not b:
        return False
    lo, hi = C.YOY_PAIR_DAYS
    return lo <= abs((a - b).days) <= hi


def zero_as_missing(rows: list, is_fin: bool) -> list:
    """Register row 35 (UNVERIFIED, off by default): a vendor 0 on a field an
    operating company always reports is a missing value, not a real zero."""
    if not C.ZERO_AS_MISSING or is_fin:
        return rows
    out = []
    for r in rows:
        r = dict(r)
        for fld in C.ZERO_AS_MISSING_FIELDS:
            if fld in r and _f(r.get(fld)) == 0:
                r[fld] = None
        # a zero cash flow on real revenue is a missing number, not a real zero
        rev = _f(r.get("revenue"))
        if _f(r.get("operatingCashFlow")) == 0 and _ok(rev) and rev != 0:
            r["operatingCashFlow"] = None
        if (_f(r.get("totalDebt")) == 0 and _f(r.get("cashAndShortTermInvestments")) == 0):
            r["totalDebt"] = r["cashAndShortTermInvestments"] = None
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# the volume line (§7 + the agreed substitutes)
# ---------------------------------------------------------------------------
def volume_line(row: dict, is_fin: bool, fin_kind: str | None) -> float:
    """Quarterly revenue; PPOP for banks/NBFCs; GWP (proxied by revenue) for
    insurers."""
    rev = _f(row.get("revenue"))
    if not is_fin or fin_kind == "insurer":
        return rev
    ie, oe = _f(row.get("interestExpense")), _f(row.get("operatingExpenses"))
    # row 13: PPOP needs all three lines. A missing line read as 0 makes that
    # quarter's PPOP jump against its neighbours.
    if not _ok(rev, ie, oe):
        return NA
    return rev - ie - oe


def fin_margin_basis(rows: list, fin_kind: str | None) -> str:
    """Row 22: choose ONE basis for a margin comparison. NIM only when every
    row compared has its inputs; otherwise operating margin for all of them -
    never a NIM subtracted from an operating margin."""
    if fin_kind in ("bank", "nbfc") and rows and all(
            _ok(_f(r.get("interestIncome")), _f(r.get("interestExpense")), _f(r.get("_assets")))
            and _f(r.get("_assets")) > 0 for r in rows):
        return "nim"
    return "op"


def fin_margin(row: dict, fin_kind: str | None, basis: str | None = None) -> float:
    """NIM proxy for lenders, inverted combined ratio for insurers - both
    expressed as an operating margin so one definition serves both.

    For a bank this captures spread compression and cost discipline together.
    For an insurer it moves opposite to the combined ratio: margin up means
    underwriting and expenses improving."""
    use_nim = (basis == "nim") if basis else (fin_kind in ("bank", "nbfc"))
    if use_nim:
        ii, ie = _f(row.get("interestIncome")), _f(row.get("interestExpense"))
        assets = _f(row.get("_assets"))
        if _ok(ii, ie, assets) and assets > 0:
            return (ii - ie) / assets
        if basis == "nim":
            return NA
    return _safe(row.get("operatingIncome"), row.get("revenue"))


def _safe(a, b):
    a, b = _f(a), _f(b)
    if not _ok(a, b) or b <= 0:
        return NA
    return a / b


# ---------------------------------------------------------------------------
# price series work
# ---------------------------------------------------------------------------
def monthly_returns(closes: list[tuple[str, float]]) -> list[float]:
    """Calendar month-end closes -> monthly returns, newest last."""
    by_month = {}
    for d, c in closes:
        by_month[d[:7]] = (d, c)            # later dates overwrite: month-end wins
    months = sorted(by_month)
    out = []
    for prev, cur in zip(months, months[1:]):
        p, c = by_month[prev][1], by_month[cur][1]
        out.append(c / p - 1.0 if p else NA)
    return out


def drawdown_episodes(closes: list[float]) -> tuple[int, float]:
    """Episodes of -30% or worse from a running peak, each closed on recovery
    to within 10% of that peak (§6 Q5). Returns (count, worst).

    Counting peak-to-trough excursions rather than every dip is the point: the
    metric is 'how often and how hard does this name break', not volatility."""
    n, worst = 0, 0.0
    peak = closes[0] if closes else NA
    in_episode, trough = False, NA
    for c in closes:
        if not math.isfinite(c):
            continue
        if not in_episode:
            peak = max(peak, c) if math.isfinite(peak) else c
            if peak > 0 and c / peak - 1.0 <= C.DRAWDOWN_EPISODE:
                in_episode, trough, n = True, c, n + 1
        else:
            trough = min(trough, c)
            if peak > 0:
                worst = min(worst, trough / peak - 1.0)
            if c >= peak * C.DRAWDOWN_RECOVERY:
                in_episode, peak = False, c
    if in_episode and math.isfinite(peak) and peak > 0 and math.isfinite(trough):
        worst = min(worst, trough / peak - 1.0)
    return n, worst


_PRICE_FIELDS = ("open", "high", "low", "close", "adjClose", "vwap", "price")


def _bar_close(r: dict) -> float:
    """Row 27: adjusted close if present and positive, else close if positive,
    else missing. A null adjClose must fall back to close, and a zero close is
    not a price."""
    for key in ("adjClose", "close"):
        v = _f(r.get(key))
        if _ok(v) and v > 0:
            return v
    return NA


def build_price_facts(ohlcv: list[dict], bench_closes: dict[str, float],
                      quote_price: float = NA) -> dict:
    """Everything P1-P6, Q5 and Darvas need out of the price series."""
    rows = [r for r in ohlcv if isinstance(r, dict) and r.get("date")]
    rows.sort(key=lambda r: str(r["date"])[:10])          # oldest first
    dates = [str(r["date"])[:10] for r in rows]
    closes = [_bar_close(r) for r in rows]
    highs = [_f(r.get("high")) for r in rows]
    lows = [_f(r.get("low")) for r in rows]
    vols = [_f(r.get("volume")) for r in rows]
    out: dict = {"price_years": len(rows) / 252.0 if rows else NA}
    if len(rows) < 30:
        return out

    finite = [c for c in closes if _ok(c)]
    out["price"] = finite[-1] if finite else NA
    out["high_252"] = max([h for h in highs[-252:] if math.isfinite(h)] or [NA])

    # row 34: a stale last bar makes every price metric a statement about the
    # past, and a last close far from the quote means one of them is wrong
    last = _pdate(dates[-1])
    out["price_age_days"] = (TODAY - last).days if last else NA
    out["price_stale"] = bool(last and (TODAY - last).days > C.STALE_PRICE_DAYS)
    qp = _f(quote_price)
    out["px_mismatch"] = bool(_ok(qp, out["price"]) and qp > 0 and
                              abs(out["price"] / qp - 1.0) > C.QUOTE_CLOSE_MAX_GAP)

    mr = monthly_returns([(d, c) for d, c in zip(dates, closes) if _ok(c)])
    # skip-month convention: the most recent month is excluded because it
    # mean-reverts, so windows end at m-2
    out["monthly_returns_12_1"] = mr[-12:-1] if len(mr) >= 12 else mr[:-1]
    out["monthly_returns_6_1"] = mr[-6:-1] if len(mr) >= 6 else mr[:-1]

    # one return per date, NaN where either close is missing - never skipped,
    # so every return stays on its own date (row 27)
    daily = [closes[i] / closes[i - 1] - 1.0 if _ok(closes[i], closes[i - 1]) else NA
             for i in range(1, len(closes))]
    out["daily_returns_252"] = [x for x in daily[-252:] if _ok(x)]

    # row 37: moves too large to be real are reported for review, not dropped
    out["implausible_moves"] = [[dates[i + 1], round(x, 4)] for i, x in enumerate(daily)
                                # inclusive: an unadjusted 2-for-1 split is exactly -50%
                                if _ok(x) and abs(x) >= C.IMPLAUSIBLE_DAILY_MOVE - 1e-9]

    # row 17: P4 over t-252 to t-21, with missing days removed BEFORE the sign
    # and the up/down shares are taken
    window = daily[-252:-21] if len(daily) > 273 else daily[:-21] or daily
    window = [x for x in window if _ok(x)]
    if window:
        up = sum(1 for x in window if x > 0) / len(window) * 100.0
        down = sum(1 for x in window if x < 0) / len(window) * 100.0
        period = (1.0 + np.array(window)).prod() - 1.0
        out["pct_up_days"], out["pct_down_days"] = up, down
        out["period_return_sign"] = 1.0 if period > 0 else (-1.0 if period < 0 else 0.0)

    # P6: the benchmark's worst 15% of days over the trailing year
    if bench_closes:
        bd = sorted(bench_closes)
        bret = {bd[i]: bench_closes[bd[i]] / bench_closes[bd[i - 1]] - 1.0
                for i in range(1, len(bd)) if bench_closes[bd[i - 1]]}
        stock = {dates[i + 1]: x for i, x in enumerate(daily) if _ok(x)}
        common = sorted(set(bret) & set(stock))[-252:]
        if common:
            cut = np.percentile([bret[d] for d in common], 15)
            out["returns_on_down_days"] = [stock[d] for d in common if bret[d] <= cut]

    n, worst = drawdown_episodes([c for c in closes if math.isfinite(c)])
    out["dd_episodes"], out["dd_worst"] = n, worst
    out["_dates"], out["_closes"] = dates, closes
    out["_highs"], out["_lows"], out["_volumes"] = highs, lows, vols
    # row 6: median dollar volume, to choose between share classes
    dv = [c * v for c, v in zip(closes[-C.DOLLAR_VOLUME_DAYS:], vols[-C.DOLLAR_VOLUME_DAYS:])
          if _ok(c, v)]
    out["dollar_volume"] = float(np.median(dv)) if dv else NA
    return out


# ---------------------------------------------------------------------------
# NTM blending
# ---------------------------------------------------------------------------
def ntm_blend(dated, field, today=None, horizon=365, parts_out=None):
    """If parts_out is a list, the (fiscal-year end, estimate, weight) triples
    used are appended to it, so the blend can be shown rather than asserted."""
    today = today or TODAY
    start, end = today, today + dt.timedelta(days=horizon)
    parts, total_w = [], 0.0
    for d, e in sorted(dated, key=lambda x: x[0]):
        v = _f(e.get(field))
        if not _ok(v):
            continue
        p_start, p_end = d - dt.timedelta(days=horizon), d
        overlap = (min(end, p_end) - max(start, p_start)).days
        if overlap <= 0:
            continue
        w = overlap / horizon
        parts.append((w, v, d))
        total_w += w
    if not parts or total_w <= 0:
        return NA, 0.0, "no usable estimate rows"
    value = sum(w * v for w, v, _ in parts) / total_w
    if parts_out is not None:
        parts_out.extend([(d.isoformat(), v, w / total_w) for w, v, d in parts])
    desc = " + ".join(f"{w / total_w:.0%} FY{d.year}" for w, _, d in parts)
    return value, min(1.0, total_w), desc


def projection_growth(rev_yoy_ttm, persistence, sector):
    """g for the CALC path: delivered TTM revenue growth, floored at 0%, capped
    at 20% except for structurally fast sectors or proven repeat growers."""
    if not _ok(_f(rev_yoy_ttm)):
        return NA
    g = max(0.0, float(rev_yoy_ttm))
    proven = _ok(_f(persistence)) and persistence >= 5 / 8
    structural = any(k in (sector or "").lower()
                     for k in ("technology", "communication services"))
    return g if (proven or structural) else min(g, 0.20)


# ---------------------------------------------------------------------------
# the fact sheet
# ---------------------------------------------------------------------------
def build_facts(symbol: str, blob: dict, bench_closes: dict, mcap_hist: dict) -> dict:
    # every statement sorted newest first, one row per date (row 8)
    inc_a = normalise_periods(_rows(blob, "income_a"))
    inc_q = normalise_periods(_rows(blob, "income_q"))
    bal_a = normalise_periods(_rows(blob, "balance_a"))
    cf_q = normalise_periods(_rows(blob, "cashflow_q"))
    prof = blob.get("profile") or [{}]
    prof = prof[0] if isinstance(prof, list) and prof else {}
    quote = blob.get("quote") or [{}]
    quote = quote[0] if isinstance(quote, list) and quote else {}

    # cadence: a half-yearly reporter's TTM is two halves, and its year-ago
    # period is two rows back, not four
    cad, cad_cf = detect_cadence(inc_q), detect_cadence(cf_q)
    N, LAG, BAND = cad["n"], cad["lag"], cad["band"]

    def ttm(rows, field, offset=0, c=cad):
        """A TTM sum only over consecutive reports - a gap blanks it."""
        if not window_ok(rows, offset, c["n"], c["band"]):
            return NA
        return _sum(rows, field, n=c["n"], offset=offset)

    sector, industry = prof.get("sector"), prof.get("industry")
    _ie, _rev = ttm(inc_q, "interestExpense"), ttm(inc_q, "revenue")
    _share = _ie / _rev if _ok(_ie, _rev) and _rev > 0 else NA
    is_fin, fin_kind = classify_financial(sector, industry, _share)

    f: dict = {"symbol": symbol, "sector": sector, "industry": industry,
               "company": prof.get("companyName"), "is_fin": is_fin,
               "fin_kind": fin_kind, "currency": prof.get("currency"),
               "country": prof.get("country")}

    # row 35 (off until verified): vendor zeros on always-reported fields
    inc_q, inc_a = zero_as_missing(inc_q, is_fin), zero_as_missing(inc_a, is_fin)
    cf_q, bal_a = zero_as_missing(cf_q, is_fin), zero_as_missing(bal_a, is_fin)

    # row 24: carry the currency check from the adapter; a mismatch with no
    # rate means the statements and the price are not in the same units
    f["fx_mismatch"] = bool(blob.get("_fx_unresolved"))

    # row 12: a market cap of zero or below is missing, not a real value
    mcap = _f(quote.get("marketCap"))
    mcap = mcap if _ok(mcap) and mcap > 0 else NA
    f["mcap"] = mcap
    f["cadence"] = cad["kind"]
    f["ttm_n"], f["yoy_lag"] = N, LAG
    # row 8: a gap inside the TTM window blanks TTM and year-on-year, and is flagged
    f["gapped"] = bool(inc_q) and not window_ok(inc_q, 0, N, BAND)

    # row 33: how old is the newest statement?
    newest = _pdate(inc_q[0].get("date")) if inc_q else None
    f["statement_age_days"] = (TODAY - newest).days if newest else NA
    f["stale_statement"] = bool(newest and (TODAY - newest).days > C.STALE_STATEMENT_DAYS)

    # ---- income / cash flow, TTM ----
    f["rev_ttm"] = ttm(inc_q, "revenue")
    f["rev_ttm_prior"] = ttm(inc_q, "revenue", offset=LAG) if pair_ok(inc_q, 0, LAG) else NA
    # Gross profit is only usable where the vendor actually reported a cost of
    # revenue. A quarter whose gross profit equals its revenue is a filled-in
    # total, not a 100% margin - see config.GP_MAX_MARGIN.
    def _gp_valid(row):
        rev, gp = _f(row.get("revenue")), _f(row.get("grossProfit"))
        if not _ok(rev, gp) or rev <= 0:
            return False
        return gp / rev < C.GP_MAX_MARGIN
    span = 2 * LAG
    gp_flags = [_gp_valid(inc_q[i]) if i < len(inc_q) else False for i in range(span)]
    f["w_gp_valid_q"] = gp_flags
    ttm_gp_ok = all(gp_flags[:N])
    present = gp_flags[:min(span, len(inc_q))]
    # row 21: no quarterly rows at all is "none", not "ok"
    f["gp_quality"] = ("none" if not present or not any(present)
                       else "ok" if all(present) else "mixed")
    f["gp_ttm"] = ttm(inc_q, "grossProfit") if ttm_gp_ok else NA
    f["ebit_ttm"] = ttm(inc_q, "operatingIncome")
    f["w_ttm_dates"] = [_date(inc_q, i) for i in range(N)]
    f["w_oi_quarters"] = [_get(inc_q, i, "operatingIncome") for i in range(N)]
    f["ni_ttm"] = ttm(inc_q, "netIncome")

    # row 28: accruals and stock comp / revenue compare income and cash-flow
    # figures, so the two statements must end on the same quarter
    d_inc = _pdate(inc_q[0].get("date")) if inc_q else None
    d_cf = _pdate(cf_q[0].get("date")) if cf_q else None
    f["cf_aligned"] = bool(d_inc and d_cf and abs((d_inc - d_cf).days) <= C.INCOME_CF_ALIGN_DAYS)
    f["cfo_ttm"] = ttm(cf_q, "operatingCashFlow", c=cad_cf) if f["cf_aligned"] else NA
    f["sbc_ttm"] = ttm(cf_q, "stockBasedCompensation", c=cad_cf) if f["cf_aligned"] else NA
    f["w_cfo_ttm"] = f["cfo_ttm"]

    # row 33 (optional): stale statements also blank the TTM-based figures
    if C.STALE_BLANK_TTM and f["stale_statement"]:
        for k in ("rev_ttm", "gp_ttm", "ebit_ttm", "ni_ttm", "cfo_ttm", "sbc_ttm"):
            f[k] = NA

    # The periodic rows behind every TTM sum, as the vendor reported them, so
    # the workings can show a TTM figure as its reports rather than a single
    # opaque total.
    def _periods(rows, field, offset=0, n=N):
        return [[_date(rows, offset + i), _get(rows, offset + i, field)] for i in range(n)]
    f["w_q"] = {
        "revenue": _periods(inc_q, "revenue"),
        "revenue_prior": _periods(inc_q, "revenue", offset=LAG),
        "grossProfit": _periods(inc_q, "grossProfit"),
        "operatingIncome": _periods(inc_q, "operatingIncome"),
        "netIncome": _periods(inc_q, "netIncome"),
        "operatingCashFlow": _periods(cf_q, "operatingCashFlow", n=cad_cf["n"]),
        "stockBasedCompensation": _periods(cf_q, "stockBasedCompensation", n=cad_cf["n"]),
    }
    f["rev_yoy_ttm"] = (f["rev_ttm"] / f["rev_ttm_prior"] - 1.0
                        if _ok(f["rev_ttm"], f["rev_ttm_prior"]) and f["rev_ttm_prior"] > 0 else NA)

    # ---- balance sheet ----
    equity = _get(bal_a, 0, "totalStockholdersEquity")
    debt, cash = _get(bal_a, 0, "totalDebt"), _get(bal_a, 0, "cashAndShortTermInvestments")
    assets = _get(bal_a, 0, "totalAssets")
    f["book"], f["assets"] = equity, assets
    # row 16: tangible book needs the goodwill line - missing is not zero
    _gw = _get(bal_a, 0, "goodwillAndIntangibleAssets")
    f["book_tangible"] = equity - _gw if _ok(equity, _gw) else NA
    f["w_total_debt"], f["w_cash"], f["w_bal_date"] = debt, cash, _date(bal_a, 0)
    f["w_goodwill"] = _get(bal_a, 0, "goodwillAndIntangibleAssets")
    f["net_debt"] = debt - cash if _ok(debt, cash) else NA
    # row 12: EV needs market cap, debt AND cash - a missing line is not zero,
    # and net debt already blanks for the same name
    f["ev"] = mcap + debt - cash if _ok(mcap, debt, cash) else NA
    f["invested_capital"] = equity + f["net_debt"] if _ok(equity, f["net_debt"]) else NA
    f["capital_ratio"] = equity / assets if _ok(equity, assets) and assets > 0 else NA
    f["assets_3y"] = _get(bal_a, 3, "totalAssets")
    f["rev_3y"] = _get(inc_a, 3, "revenue")
    # row 29: bloat and dilution compare like with like - FY0 against FY-3 on
    # annual statements for both sides
    f["rev_a0"] = _get(inc_a, 0, "revenue")
    f["shares_a0"] = _get(inc_a, 0, "weightedAverageShsOutDil")
    f["w_date_bal0"], f["w_date_bal3"] = _date(bal_a, 0), _date(bal_a, 3)

    # ---- tax, NOPAT and their 3-year deltas ----
    # Handbook §6: NOPAT uses the company's OWN effective rate, clamped 0-35%,
    # and each year is taxed at the rate that applied in that year. Note the
    # consequence: because ROIIC differences two NOPATs, a genuine change in a
    # company's tax rate moves ROIIC even when EBIT is flat. That is the
    # handbook's arithmetic, not an error - it reads a permanently lower tax
    # bill as a real improvement in what the capital returns.
    pre, tax = _get(inc_a, 0, "incomeBeforeTax"), _get(inc_a, 0, "incomeTaxExpense")
    f["effective_tax_rate"] = tax / pre if _ok(pre, tax) and pre > 0 else NA

    def rate_at(i):
        pre_i, tax_i = _get(inc_a, i, "incomeBeforeTax"), _get(inc_a, i, "incomeTaxExpense")
        r = tax_i / pre_i if _ok(pre_i, tax_i) and pre_i > 0 else NA
        return M_effective_tax(r)

    def nopat_at(i):
        e = _get(inc_a, i, "operatingIncome")
        return e * (1 - rate_at(i)) if _ok(e) else NA

    def ic_at(i):
        eq = _get(bal_a, i, "totalStockholdersEquity")
        d, c = _get(bal_a, i, "totalDebt"), _get(bal_a, i, "cashAndShortTermInvestments")
        return eq + (d - c) if _ok(eq, d, c) else NA

    # exposed so the calculation can be shown with its inputs, not just its result
    f["w_pretax"], f["w_tax_exp"] = pre, tax
    f["w_ebit_a0"], f["w_ebit_a3"] = _get(inc_a, 0, "operatingIncome"), _get(inc_a, 3, "operatingIncome")
    f["w_rate_a0"], f["w_rate_a3"] = rate_at(0), rate_at(3)
    f["w_nopat_0"], f["w_nopat_3"] = nopat_at(0), nopat_at(3)
    f["w_ic_0"], f["w_ic_3"] = ic_at(0), ic_at(3)
    f["w_date_a0"], f["w_date_a3"] = _date(inc_a, 0), _date(inc_a, 3)
    f["d_nopat_3y"] = nopat_at(0) - nopat_at(3) if _ok(nopat_at(0), nopat_at(3)) else NA
    f["d_ic_3y"] = ic_at(0) - ic_at(3) if _ok(ic_at(0), ic_at(3)) else NA
    ni0, ni3 = _get(inc_a, 0, "netIncome"), _get(inc_a, 3, "netIncome")
    bk0, bk3 = _get(bal_a, 0, "totalStockholdersEquity"), _get(bal_a, 3, "totalStockholdersEquity")
    f["d_ni_3y"] = ni0 - ni3 if _ok(ni0, ni3) else NA
    f["d_book_3y"] = bk0 - bk3 if _ok(bk0, bk3) else NA
    f["w_ni_0"], f["w_ni_3"], f["w_book_0"], f["w_book_3"] = ni0, ni3, bk0, bk3

    # ---- 5-year average margins / ROE ----
    margins, roes = [], []
    f["w_margin_years"], f["w_roe_years"] = [], []
    for i in range(5):
        r, n = _get(inc_a, i, "revenue"), _get(inc_a, i, "netIncome")
        if _ok(r, n) and r > 0:
            margins.append(n / r)
            f["w_margin_years"].append([_date(inc_a, i), n, r, n / r])
        eq = _get(bal_a, i, "totalStockholdersEquity")
        if _ok(n, eq) and eq > 0:
            roes.append(n / eq)
            f["w_roe_years"].append([_date(inc_a, i), n, eq, n / eq])
    f["w_margins"] = [round(x, 6) for x in margins]
    f["w_roes"] = [round(x, 6) for x in roes]
    f["net_margin_5y_avg"] = float(np.mean(margins)) if margins else NA
    f["margin_years"] = len(margins)
    f["roe_5y_avg"] = float(np.mean(roes)) if roes else NA
    f["roe_years"] = len(roes)              # row 20: every 5-year average is held to a minimum

    # ---- shares ----
    f["shares_now"] = _get(inc_q, 0, "weightedAverageShsOutDil")
    f["shares_3y"] = _get(inc_a, 3, "weightedAverageShsOutDil")

    # ---- forward estimates ----
    # one row per fiscal-year date, sorted by date only: two rows on the same
    # date would otherwise be compared as dicts and fail the whole name
    dated, seen_dates = [], set()
    for e in _rows(blob, "estimates"):
        try:
            d_est = dt.date.fromisoformat(str(e.get("date"))[:10])
        except (TypeError, ValueError):
            continue
        if d_est in seen_dates:
            continue
        seen_dates.add(d_est)
        dated.append((d_est, e))
    dated.sort(key=lambda x: x[0])
    f["w_ntm_parts"] = []
    eps, cov, desc = ntm_blend(dated, "estimatedEpsAvg", parts_out=f["w_ntm_parts"])
    # row 23: the fallback may only use a fiscal year that has not yet ended
    cutoff = TODAY + dt.timedelta(days=C.NTM_FALLBACK_MIN_DAYS_AHEAD)
    future = [(d, e) for d, e in dated if d > cutoff]
    if _ok(eps) and cov >= 0.80:
        f["ntm_eps"], f["fwd_basis"] = eps, "ntm"
        used = [e for d, e in dated if d.isoformat() in {p[0] for p in f["w_ntm_parts"]}]
    else:
        f["ntm_eps"] = _f(future[0][1].get("estimatedEpsAvg")) if future else NA
        f["fwd_basis"] = "fy_fallback" if future else "none"
        used = [future[0][1]] if future else []
    # row 36 (UNVERIFIED - runs only once the field name is confirmed): too few
    # analysts behind a consensus is treated as no consensus -> the CALC path
    f["analyst_count"] = NA
    if C.ANALYST_COUNT_FIELD and used:
        counts = [_f(e.get(C.ANALYST_COUNT_FIELD)) for e in used]
        f["analyst_count"] = min(c for c in counts if _ok(c)) if any(_ok(c) for c in counts) else NA
        if not _ok(f["analyst_count"]) or f["analyst_count"] < C.MIN_ANALYSTS:
            f["ntm_eps"], f["fwd_basis"] = NA, "none"
            f["thin_consensus"] = True
    f["ntm_coverage"], f["ntm_blend"] = cov, desc
    # forward revenue, blended the same way - context only, never ranked
    rev_ntm, rev_cov, _ = ntm_blend(dated, "estimatedRevenueAvg")
    if _ok(rev_ntm) and rev_cov >= 0.80:
        f["fwd_rev"], f["fwd_rev_basis"] = rev_ntm, "ntm"
    else:
        future_r = [(d, e) for d, e in dated if d > cutoff]
        f["fwd_rev"] = _f(future_r[0][1].get("estimatedRevenueAvg")) if future_r else NA
        f["fwd_rev_basis"] = "fy_fallback" if future_r else "none"

    # ---- volume line, persistence, acceleration ----
    for r in inc_q:
        r["_assets"] = assets
    vols = [volume_line(r, is_fin, fin_kind) for r in inc_q]
    f["volume_quarters"] = sum(1 for v in vols if math.isfinite(v))
    # row 8: each period against the same period a year earlier - LAG rows
    # back - and only where the two really are a year apart
    yoy = [vols[i] / vols[i + LAG] - 1.0 if i + LAG < len(vols)
           and _ok(vols[i], vols[i + LAG]) and vols[i + LAG] > 0 and pair_ok(inc_q, i, i + LAG)
           else NA for i in range(len(vols))]
    f["vol_yoy_q0"] = yoy[0] if yoy else NA
    f["vol_yoy_q1"] = yoy[1] if len(yoy) > 1 else NA
    # persistence over two years of reports: 8 quarters or 4 halves
    win = 2 * LAG
    f["persistence_window"], f["persistence_min"] = win, LAG
    hits = [1 for v in yoy[:win] if math.isfinite(v) and v > 0]
    f["persistence_quarters"] = sum(1 for v in yoy[:win] if math.isfinite(v))
    f["persistence_hits"] = len(hits)
    f["b2_min_periods"] = LAG + 2          # two consecutive year-on-year readings
    # 3-year CAGR of the volume line, taken from the ANNUAL statements. Doing it
    # off quarterly data would need 16 quarters and only 12 are fetched, which
    # silently blanked B1 for every non-financial.
    for r in inc_a:
        r["_assets"] = assets
    ann_vol = [volume_line(r, is_fin, fin_kind) for r in inc_a]
    v_now = ann_vol[0] if ann_vol else NA
    v_3y = ann_vol[3] if len(ann_vol) > 3 else NA
    f["vol_cagr_3y"] = ((v_now / v_3y) ** (1 / 3) - 1.0
                        if _ok(v_now, v_3y) and v_3y > 0 and v_now > 0 else NA)
    # the quarter-level and year-level inputs behind the accelerations
    f["w_vol_a0"], f["w_vol_a3"] = v_now, v_3y
    f["w_vol_q"] = [vols[i] if i < len(vols) else NA for i in range(LAG + 2)]
    f["w_vol_q12"] = [[_date(inc_q, i), vols[i] if i < len(vols) else NA] for i in range(3 * LAG)]
    f["w_date_q"] = [_date(inc_q, i) for i in range(LAG + 2)]
    f["w_vol_yoy"] = [yoy[i] if i < len(yoy) else NA for i in range(2 * LAG)]
    f["w_vol_label"] = ("pre-provision profit" if is_fin and fin_kind != "insurer"
                        else "revenue")

    # ---- margin direction ----
    # the period LAG rows back is the same period a year earlier; the pair must
    # really be a year apart (row 8)
    yr_pair = pair_ok(inc_q, 0, LAG)
    f["w_lag"] = LAG
    if is_fin:
        pair_rows = [inc_q[0], inc_q[LAG]] if len(inc_q) > LAG else []
        basis = fin_margin_basis(pair_rows, fin_kind)       # row 22: one basis for both
        f["fin_margin_basis"] = basis
        f["fin_margin_q0"] = fin_margin(inc_q[0], fin_kind, basis) if pair_rows and yr_pair else NA
        f["fin_margin_q4"] = fin_margin(inc_q[LAG], fin_kind, basis) if pair_rows and yr_pair else NA
    else:
        # comparing a filled-in quarter with a real one manufactures a huge
        # swing, so both ends must carry a genuine cost of revenue
        both = gp_flags[0] and (gp_flags[LAG] if len(gp_flags) > LAG else False) and yr_pair
        f["gm_q0"] = _safe(_get(inc_q, 0, "grossProfit"), _get(inc_q, 0, "revenue")) if both else NA
        f["gm_q4"] = _safe(_get(inc_q, LAG, "grossProfit"), _get(inc_q, LAG, "revenue")) if both else NA
    # True only when a RANKED metric was set aside - a filled-in quarter outside
    # both the TTM window and the year-on-year comparison changes nothing
    f["gp_set_aside"] = (not is_fin) and (not ttm_gp_ok or not (gp_flags[0] and
                         (gp_flags[LAG] if len(gp_flags) > LAG else False)))
    f["w_gp_q0"], f["w_rev_q0"] = _get(inc_q, 0, "grossProfit"), _get(inc_q, 0, "revenue")
    f["w_gp_q4"], f["w_rev_q4"] = _get(inc_q, LAG, "grossProfit"), _get(inc_q, LAG, "revenue")
    f["w_oi_q0"], f["w_oi_q4"] = _get(inc_q, 0, "operatingIncome"), _get(inc_q, LAG, "operatingIncome")
    # a lender's margin is interest spread on assets, so show those lines for banks
    f["w_ii_q0"], f["w_ie_q0"] = _get(inc_q, 0, "interestIncome"), _get(inc_q, 0, "interestExpense")
    f["w_ii_q4"], f["w_ie_q4"] = _get(inc_q, LAG, "interestIncome"), _get(inc_q, LAG, "interestExpense")

    # ---- growth, carried as CONTEXT only -------------------------------
    # This model has no growth pillar: growth enters value through the forward
    # earnings yield (commitment 2), and persistence sits in quality. These are
    # kept so the analyst can still see the growth picture -
    # unscored, never ranked.
    f["ctx_rev_cagr_3y"] = ((_get(inc_a, 0, "revenue") / _get(inc_a, 3, "revenue")) ** (1 / 3) - 1
                            if _ok(_get(inc_a, 0, "revenue"), _get(inc_a, 3, "revenue"))
                            and _get(inc_a, 3, "revenue") > 0 else NA)
    f["ctx_rev_yoy_ttm"] = f["rev_yoy_ttm"]
    f["ctx_rev_accel"] = (f["ctx_rev_yoy_ttm"] - f["ctx_rev_cagr_3y"]
                          if _ok(f["ctx_rev_yoy_ttm"], f["ctx_rev_cagr_3y"]) else NA)
    gp_prior = (ttm(inc_q, "grossProfit", offset=LAG)
                if all(gp_flags[LAG:LAG + N]) and pair_ok(inc_q, 0, LAG) else NA)
    f["ctx_gp_growth_ttm"] = (f["gp_ttm"] / gp_prior - 1
                              if _ok(f["gp_ttm"], gp_prior) and gp_prior > 0 else NA)
    f["ctx_fwd_rev_growth"] = (f["fwd_rev"] / f["rev_ttm"] - 1
                               if _ok(_f(f.get("fwd_rev")), f["rev_ttm"]) and f["rev_ttm"] > 0 else NA)
    eps_ttm = ttm(inc_q, "epsdiluted")
    f["ctx_fwd_eps_growth"] = (f["ntm_eps"] / eps_ttm - 1
                               if _ok(_f(f.get("ntm_eps")), eps_ttm) and eps_ttm > 0
                               and _f(f.get("ntm_eps")) > 0 else NA)
    f["ctx_persistence"] = (f["persistence_hits"] / f["persistence_quarters"]
                            if f["persistence_quarters"] else NA)
    f["ctx_gross_margin"] = _safe(f["gp_ttm"], f["rev_ttm"])
    _cfo = ttm(cf_q, "operatingCashFlow", c=cad_cf) if f["cf_aligned"] else NA
    _capex = ttm(cf_q, "capitalExpenditure", c=cad_cf) if f["cf_aligned"] else NA
    f["ctx_fcf_margin"] = _safe(_cfo + _capex, f["rev_ttm"]) if _ok(_cfo, _capex) else NA

    f["g_projection"] = projection_growth(f["rev_yoy_ttm"],
                                          f["persistence_hits"] / f["persistence_quarters"]
                                          if f["persistence_quarters"] else NA, sector)

    # ---- own-history medians (V5) ----
    ratios = []
    f["w_hist_points"] = []           # each year's point, for the workings
    for i in range(5):
        d = _date(inc_a, i)
        if not d:
            continue
        m = mcap_hist.get(d) or _nearest(mcap_hist, d)
        if not _ok(_f(m)):
            continue
        if is_fin:
            eq = _get(bal_a, i, "totalStockholdersEquity")
            if _ok(eq) and eq > 0:
                ratios.append(m / eq)
                f["w_hist_points"].append([d, m, eq, None, None, m / eq])
        else:
            rev = _get(inc_a, i, "revenue")
            dd, cc = _get(bal_a, i, "totalDebt"), _get(bal_a, i, "cashAndShortTermInvestments")
            if _ok(rev, dd, cc) and rev > 0:
                ratios.append((m + dd - cc) / rev)
                f["w_hist_points"].append([d, m, dd, cc, rev, (m + dd - cc) / rev])
    med = float(statistics.median(ratios)) if len(ratios) >= 3 else NA
    f["pb_median_hist" if is_fin else "ev_sales_median_hist"] = med

    # ---- price-derived ----
    # a minor-unit line (GBp, ZAc, ILA) is scaled to the major unit, the unit
    # its market cap and converted statements are in; returns are unchanged
    scale = _f(blob.get("_price_unit_scale"))
    scale = scale if _ok(scale) and scale > 0 else 1.0
    f["price_unit_scale"] = scale
    bars = _rows(blob, "ohlcv")
    if scale != 1.0:
        bars = [{k: (v * scale if k in _PRICE_FIELDS and _ok(_f(v)) else v) for k, v in r.items()}
                if isinstance(r, dict) else r for r in bars]
    f.update(build_price_facts(bars, bench_closes, _f(quote.get("price")) * scale))

    # turnover in one currency (config.TURNOVER_CURRENCY), so share classes and
    # cross-listings in different currencies are compared like with like
    rate = _f(blob.get("_turnover_rate"))
    if not _ok(rate) and str(f.get("currency") or "").upper() == C.TURNOVER_CURRENCY:
        rate = 1.0
    dv = _f(f.get("dollar_volume"))
    f["dollar_volume"] = dv * rate if _ok(dv, rate) else NA

    # per-share consensus on the traded line's share basis (an ADR may carry
    # several ordinary shares): restate EPS per traded unit when the reported
    # share count and market cap / price disagree beyond the tolerance
    units = _safe(f.get("mcap"), f.get("price"))
    basis = _safe(f.get("shares_now"), units) if _ok(_f(units)) and units > 0 else NA
    f["share_basis"] = basis
    tol = C.SHARE_BASIS_TOLERANCE
    if _ok(_f(basis)) and basis > 0 and not (1 / (1 + tol) <= basis <= 1 + tol):
        if _ok(_f(f.get("ntm_eps"))):
            f["ntm_eps"] = f["ntm_eps"] * basis
            # the workings re-add the blend, so its parts move to the same basis
            f["w_ntm_parts"] = [(d, v * basis, wt) for d, v, wt in f.get("w_ntm_parts") or []]
        f["share_basis_restated"] = True
    else:
        f["share_basis_restated"] = False
    return f


def _nearest(series: dict, target: str, max_days: int = 10):
    """Market cap on the nearest trading day at or before a statement date."""
    if not series:
        return None
    try:
        t = dt.date.fromisoformat(target)
    except ValueError:
        return None
    best, best_gap = None, max_days + 1
    for d, v in series.items():
        try:
            gap = (t - dt.date.fromisoformat(d)).days
        except ValueError:
            continue
        if 0 <= gap < best_gap:
            best, best_gap = v, gap
    return best
