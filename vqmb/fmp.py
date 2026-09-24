"""
VQMB — the FMP /stable/ data layer.

Fetches everything the model reads from Financial Modeling Prep and normalises
each response into the shape the fact sheet (facts.py) expects. Nothing in here
changes a number the model computes: field aliases only add the canonical name
when FMP sends an older or newer spelling, and the currency step puts the
statements into the same currency as the share price.

The caller's own FMP API key is passed into every function; nothing is read
from the environment here.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger("vqmb")

STABLE_BASE = "https://financialmodelingprep.com/stable"

MAX_RETRIES = 4
RETRY_BASE_SLEEP = 1.5

# Raw vendor responses may be cached on disk (opt-in, see fetch_ticker). The
# fundamentals only change on an earnings report, so a cached copy is reused for
# up to a day; market-cap history only feeds V5's five-year median and is reused
# for a week. Prices are always fetched fresh.
FUNDAMENTALS_CACHE_DAYS = 1.0
MCAP_CACHE_DAYS = 7

# ---------------------------------------------------------------------------
# Stable endpoint map - only the seven endpoints the model reads. Every path is
# query-param style: ?symbol=...
# ---------------------------------------------------------------------------
STABLE_ENDPOINTS = {
    "profile":     ("/profile",                  {"symbol": "{t}"}),
    "quote":       ("/quote",                    {"symbol": "{t}"}),
    "income_a":    ("/income-statement",         {"symbol": "{t}", "period": "annual", "limit": 6}),
    "income_q":    ("/income-statement",         {"symbol": "{t}", "period": "quarter", "limit": 12}),
    "balance_a":   ("/balance-sheet-statement",  {"symbol": "{t}", "period": "annual", "limit": 6}),
    "cashflow_q":  ("/cash-flow-statement",      {"symbol": "{t}", "period": "quarter", "limit": 8}),
    "estimates":   ("/analyst-estimates",        {"symbol": "{t}", "period": "annual", "limit": 6}),
}

INDEX_PATHS = {
    "sp500": "/sp500-constituent",
    "nasdaq": "/nasdaq-constituent",
    "dowjones": "/dowjones-constituent",
}

# ---------------------------------------------------------------------------
# Field aliases. FMP renamed a handful of fields between API generations.
# Each entry: canonical name the model reads  <-  list of names FMP may send.
# Applied per row; the first alias that exists wins. Unknown/missing stays
# missing -> the model's own NaN handling takes over. No invented data.
# ---------------------------------------------------------------------------
ALIASES = {
    "profile": {
        "lastDiv": ["lastDiv", "lastDividend"],
        "companyName": ["companyName", "name"],
        "sector": ["sector"],
        "industry": ["industry"],
        "country": ["country"],
    },
    "quote": {
        "price": ["price", "close"],
        "marketCap": ["marketCap", "marketCapitalization"],
    },
    "income": {
        "revenue": ["revenue"],
        "grossProfit": ["grossProfit"],
        "netIncome": ["netIncome"],
        "operatingIncome": ["operatingIncome"],
        "epsdiluted": ["epsdiluted", "epsDiluted", "epsdilutedRatio"],
        "researchAndDevelopmentExpenses": ["researchAndDevelopmentExpenses"],
        "sellingGeneralAndAdministrativeExpenses": [
            "sellingGeneralAndAdministrativeExpenses",
            "generalAndAdministrativeExpenses",
        ],
        "weightedAverageShsOutDil": ["weightedAverageShsOutDil", "weightedAverageShsOutstandingDil"],
    },
    "balance": {
        "totalAssets": ["totalAssets"],
        "totalDebt": ["totalDebt"],
        "totalStockholdersEquity": ["totalStockholdersEquity", "totalEquity"],
        "cashAndShortTermInvestments": ["cashAndShortTermInvestments"],
        "accountPayables": ["accountPayables", "accountsPayables", "accountPayable"],
        "netReceivables": ["netReceivables", "accountsReceivables"],
    },
    "cashflow": {
        "operatingCashFlow": [
            "operatingCashFlow",
            "netCashProvidedByOperatingActivities",
            "netCashProvidedByOperatingActivites",   # FMP's own historical typo
        ],
        "capitalExpenditure": ["capitalExpenditure", "investmentsInPropertyPlantAndEquipment"],
        "freeCashFlow": ["freeCashFlow"],
        "stockBasedCompensation": ["stockBasedCompensation"],
        "dividendsPaid": ["dividendsPaid", "netDividendsPaid", "commonDividendsPaid"],
        "commonStockRepurchased": ["commonStockRepurchased", "netCommonStockRepurchased",
                                   "commonStockRepurchase"],
        "changeInWorkingCapital": ["changeInWorkingCapital"],
    },
    "estimates": {
        "estimatedRevenueAvg": ["estimatedRevenueAvg", "revenueAvg"],
        "estimatedEpsAvg": ["estimatedEpsAvg", "epsAvg"],
        "date": ["date"],
    },
}

_ROW_KIND = {
    "profile": "profile", "quote": "quote",
    "income_a": "income", "income_q": "income",
    "balance_a": "balance",
    "cashflow_q": "cashflow",
    "estimates": "estimates",
}


def _alias_row(row: dict, kind: str) -> dict:
    """Add canonical keys to a row without deleting anything FMP sent."""
    if not isinstance(row, dict):
        return row
    spec = ALIASES.get(kind)
    if not spec:
        return row
    out = dict(row)
    for canonical, candidates in spec.items():
        if out.get(canonical) is not None:
            continue
        for c in candidates:
            if row.get(c) is not None:
                out[canonical] = row[c]
                break
    return out


def _alias(payload, kind: str):
    if isinstance(payload, list):
        return [_alias_row(r, kind) for r in payload]
    if isinstance(payload, dict):
        return _alias_row(payload, kind)
    return payload


# ---------------------------------------------------------------------------
# HTTP with retry/backoff
# ---------------------------------------------------------------------------
def _get(url: str, params: dict, api_key: str):
    p = dict(params)
    p["apikey"] = api_key
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=p, timeout=40)
            if r.status_code == 429:
                time.sleep(RETRY_BASE_SLEEP * (2 ** attempt))
                last = RuntimeError("429 rate limited")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:      # network blip, 5xx, bad JSON
            last = e
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_BASE_SLEEP * (2 ** attempt))
    raise last


def _fill(params: dict, ticker: str) -> dict:
    ctx = {"t": ticker}
    out = {}
    for k, v in params.items():
        out[k] = v.format(**ctx) if isinstance(v, str) else v
    return out


# ---------------------------------------------------------------------------
# CURRENCY NORMALISATION
# ---------------------------------------------------------------------------
# FMP reports financial statements in the company's own reporting currency and
# quotes the price in the currency of the listing. For most companies these are
# the same. For a company that is cross-listed and files abroad they are not,
# and every yield the model computes would divide one currency by another.
#
# The statements are converted INTO the quote currency. Flows (income, cash
# flow) use the average rate across their period; a balance sheet uses the
# closing rate on its date; forward estimates use today's spot. Every ratio
# computed inside the statements - growth rates, margins, returns on capital -
# is unchanged by a constant multiplier; only the statement-versus-market
# comparisons move, which are exactly the ones that were wrong.
#
# Blobs carry a schema marker so cached copies written under a different layout
# are refetched rather than silently reused.
CACHE_SCHEMA = 2

# Keys inside a statement row that are NOT amounts of money and must never be
# scaled: share counts, years, identifiers, and anything already a ratio.
_NON_MONETARY_EXACT = {
    "date", "symbol", "reportedCurrency", "cik", "filingDate", "acceptedDate",
    "fiscalYear", "calendarYear", "period", "link", "finalLink",
    "weightedAverageShsOut", "weightedAverageShsOutDil",
    "weightedAverageShsOutstanding", "weightedAverageShsOutstandingDil",
    "sharesOutstanding", "commonStockSharesOutstanding", "numberOfShares",
}
# Substrings are matched against the whole key, so they must not appear inside a
# genuine money field. "count" is deliberately absent: it matches accountPayables.
_NON_MONETARY_SUBSTR = ("ratio", "margin", "percent", "growth", "yield",
                        "pershare", "persharediluted")


def _is_monetary(key: str) -> bool:
    if key in _NON_MONETARY_EXACT:
        return False
    k = key.lower()
    return not any(sub in k for sub in _NON_MONETARY_SUBSTR)


_FX_CACHE = {}
_FX_SERIES_CACHE = {}


def fx_series(base: str, quote: str, api_key: str):
    """Daily history of `quote` per one `base`, as {date: rate}.

    Statements are converted at the rate that was true on the statement's own
    date, not today's. Using one current rate would restate several-year-old
    accounts at today's exchange rate.
    """
    if base == quote:
        return {}
    key = (base, quote)
    if key in _FX_SERIES_CACHE:
        return _FX_SERIES_CACHE[key]

    today = dt.date.today()
    params = {"from": (today - dt.timedelta(days=2600)).isoformat(),
              "to": today.isoformat()}
    out = {}
    for sym, invert in ((f"{base}{quote}", False), (f"{quote}{base}", True)):
        try:
            rows = _get(STABLE_BASE + "/historical-price-eod/light",
                        dict(params, symbol=sym), api_key)
        except Exception:
            continue
        if isinstance(rows, dict):
            rows = rows.get("historical") or []
        for r in rows if isinstance(rows, list) else []:
            if not isinstance(r, dict):
                continue
            d = str(r.get("date"))[:10]
            v = r.get("close", r.get("price", r.get("adjClose")))
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if d and v > 0:
                out[d] = (1.0 / v) if invert else v
        if out:
            break

    _FX_SERIES_CACHE[key] = out
    return out


def rate_avg(series: dict, spot, as_of: str, days: int):
    """Mean daily rate over the `days` ending on `as_of`.

    Income and cash-flow figures are flows earned across a period, so they are
    translated at the period's AVERAGE rate; a balance sheet is a position on
    one date and uses that date's closing rate (IAS 21).
    """
    if not series or not as_of:
        return rate_on(series, spot, as_of)
    try:
        end = dt.date.fromisoformat(as_of)
    except ValueError:
        return rate_on(series, spot, as_of)
    start = (end - dt.timedelta(days=days)).isoformat()
    window = [v for d, v in series.items() if start < d <= as_of]
    if len(window) < max(5, days // 20):        # too thin to be an average
        return rate_on(series, spot, as_of)
    return sum(window) / len(window)


def rate_on(series: dict, spot, as_of: str):
    """Rate on `as_of`, else the most recent earlier date. Falls back to spot
    for dates before the series starts (or for forward-looking estimates)."""
    if not series:
        return spot
    if as_of and as_of in series:
        return series[as_of]
    if as_of:
        earlier = [d for d in series if d <= as_of]
        if earlier:
            return series[max(earlier)]
    return spot


def fx_rate(base: str, quote: str, api_key: str):
    """Units of `quote` per one unit of `base`, e.g. fx_rate('USD','INR') -> ~84."""
    if base == quote:
        return 1.0
    key = (base, quote)
    if key in _FX_CACHE:
        return _FX_CACHE[key]

    def _read(payload):
        row = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(row, dict):
            return None
        for f in ("price", "rate", "bid", "close", "previousClose"):
            v = row.get(f)
            if v:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    rate = None
    for path, sym, invert in (("/forex-quote", f"{base}{quote}", False),
                              ("/quote", f"{base}{quote}", False),
                              ("/forex-quote", f"{quote}{base}", True),
                              ("/quote", f"{quote}{base}", True)):
        try:
            v = _read(_get(STABLE_BASE + path, {"symbol": sym}, api_key))
        except Exception:
            v = None
        if v and v > 0:
            rate = (1.0 / v) if invert else v
            break

    _FX_CACHE[key] = rate          # cache the miss too; do not retry per ticker
    return rate


# How each statement type is translated:
#   flow over a period -> average rate across that period
#   position on a date -> closing rate on that date
#   dated in the future -> current spot (no historical rate exists yet)
CONV_BASIS = {
    "income_a":   ("avg", 365),
    "income_q":   ("avg", 92),
    "cashflow_q": ("avg", 92),
    "balance_a":  ("close", 0),     # a position, not a flow
    "estimates":  ("spot", 0),
}


def _convert_rows(rows, series, spot, basis=("close", 0)):
    """Convert one statement list, each row on its own date and basis."""
    if not isinstance(rows, list):
        return rows, []
    kind, days = basis
    used = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = str(r.get("date"))[:10] if r.get("date") else None
        if kind == "spot":
            mult = spot
        elif kind == "avg":
            mult = rate_avg(series, spot, d, days)
        else:
            mult = rate_on(series, spot, d)
        if not mult:
            continue
        for k, v in list(r.items()):
            if isinstance(v, (int, float)) and not isinstance(v, bool) and _is_monetary(k):
                r[k] = v * mult
        r["_fxRate"] = mult            # provenance: what this row was converted at
        r["_fxBasis"] = kind
        used.append((d, mult))
    return rows, used


def normalise_currency(blob: dict, ticker: str, api_key: str) -> dict:
    """Put the statements into the same currency as the price. Records what it
    did in blob['_fx']."""
    inc = blob.get("income_a") or blob.get("income_q") or []
    reported = None
    for r in (inc if isinstance(inc, list) else []):
        if isinstance(r, dict) and r.get("reportedCurrency"):
            reported = str(r["reportedCurrency"]).upper()
            break
    prof = blob.get("profile")
    prof = prof[0] if isinstance(prof, list) and prof else (prof if isinstance(prof, dict) else {})
    quoted = str((prof or {}).get("currency") or "").upper() or None

    info = {"reported_currency": reported, "quote_currency": quoted,
            "rate": 1.0, "applied": False, "status": "same currency"}

    if not reported or not quoted:
        info["status"] = "unknown — currency field missing"
        blob["_fx"] = info
        return blob
    if reported == quoted:
        blob["_fx"] = info
        return blob

    spot = fx_rate(reported, quoted, api_key)
    if not spot:
        info["status"] = (f"MISMATCH {reported} vs {quoted} — no FX rate available, "
                          f"statements left unconverted")
        # register row 24: the facts read this, raise the FX flag and set the
        # five value lenses aside rather than rank unconverted numbers
        blob["_fx_unresolved"] = True
        log.warning("%s: statements in %s, price in %s, no FX rate — value metrics set aside",
                    ticker, reported, quoted)
        blob["_fx"] = info
        return blob

    series = fx_series(reported, quoted, api_key)
    all_used = []
    for key, basis in CONV_BASIS.items():
        rows, used = _convert_rows(blob.get(key), series, spot, basis=basis)
        blob[key] = rows
        all_used += used

    rates = [m for _, m in all_used if m]
    basis = ("period rates (flows at period averages, balance sheet at closing)"
             if series else "current spot (no FX history available)")
    info.update(rate=spot, applied=True, basis=basis,
                n_days_history=len(series),
                rate_min=min(rates) if rates else None,
                rate_max=max(rates) if rates else None,
                status=(f"converted {reported} -> {quoted} using {basis}"
                        + (f", {min(rates):.2f}-{max(rates):.2f} across periods"
                           if rates and len(set(rates)) > 1 else f" at {spot:.4f}")))
    log.info("%s: statements %s -> %s, %s", ticker, reported, quoted, basis)
    blob["_fx"] = info
    return blob


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------
def fetch_ticker(ticker: str, api_key: str, cache_dir: Path | None = None,
                 refresh: bool = False) -> dict:
    """Every statement endpoint for one name, aliased and currency-normalised.

    cache_dir: optional folder for raw responses. A cached copy is reused for up
    to a day unless `refresh` is set. A copy with ANY failed endpoint (or an
    unresolved currency) is never reused - it is refetched in full, so one bad
    fetch cannot ride the cache into later runs (register row 25).
    """
    cpath = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath = cache_dir / f"{ticker}.json"
        cached = None
        if cpath.exists():
            try:
                cached = json.loads(cpath.read_text())
            except Exception:
                cached = None
        if cached is not None and cached.get("_schema") != CACHE_SCHEMA:
            cached = None
        if cached is not None and (cached.get("_failed") or cached.get("_fx_unresolved")):
            cached = None
        if cached is not None and not refresh:
            if (time.time() - cpath.stat().st_mtime) / 86400 < FUNDAMENTALS_CACHE_DAYS:
                return cached

    blob, failed = {}, []
    for name, (path, params) in STABLE_ENDPOINTS.items():
        try:
            raw = _get(STABLE_BASE + path, _fill(params, ticker), api_key)
            blob[name] = _alias(raw, _ROW_KIND[name])
        except Exception as e:
            blob[name] = None
            failed.append(name)
            log.warning("%s %s: %s", ticker, name, e)
        time.sleep(0.05)
    blob = normalise_currency(blob, ticker, api_key)
    blob["_failed"] = failed          # recorded, so the cache check above can refuse it
    blob["_schema"] = CACHE_SCHEMA
    if cpath is not None:
        cpath.write_text(json.dumps(blob))
    return blob


def fetch_extra(symbol: str, api_key: str, years: int, cache_dir: Path | None = None) -> dict:
    """OHLCV and historical market cap - the two series the statement blob lacks.

    Prices are always fetched fresh. Market-cap history may be cached for a week
    when cache_dir is given - it only feeds V5's five-year median."""
    today = dt.date.today()
    frm = (today - dt.timedelta(days=int(years * 365.25) + 30)).isoformat()
    out = {"ohlcv": [], "mcap_hist": {}}
    try:
        rows = _get(STABLE_BASE + "/historical-price-eod/full",
                    {"symbol": symbol, "from": frm, "to": today.isoformat()}, api_key)
        out["ohlcv"] = rows if isinstance(rows, list) else rows.get("historical", [])
    except Exception as e:
        log.warning("%s ohlcv: %s", symbol, e)
    cache = None
    if cache_dir is not None:
        d = Path(cache_dir) / "mcap"
        d.mkdir(parents=True, exist_ok=True)
        cache = d / f"{symbol}.json"
        if cache.exists() and (dt.datetime.now().timestamp() - cache.stat().st_mtime) \
                < MCAP_CACHE_DAYS * 86400:
            try:
                out["mcap_hist"] = json.loads(cache.read_text())
                return out
            except Exception:
                pass
    try:
        rows = _get(STABLE_BASE + "/historical-market-capitalization",
                    {"symbol": symbol, "from": frm, "to": today.isoformat(),
                     "limit": 5000}, api_key)
        for r in rows if isinstance(rows, list) else []:
            d = str(r.get("date"))[:10]
            v = r.get("marketCap")
            if d and v:
                out["mcap_hist"][d] = float(v)
        # row 26: never cache an empty history - it would blank V5 for a week
        if cache is not None and out["mcap_hist"]:
            cache.write_text(json.dumps(out["mcap_hist"]))
    except Exception as e:
        log.warning("%s mcap history: %s", symbol, e)
    return out


def benchmark_closes(api_key: str, years: int, symbol: str = "SPY") -> dict:
    """{date: close} for the benchmark, used by P6."""
    today = dt.date.today()
    frm = (today - dt.timedelta(days=int(years * 365.25) + 30)).isoformat()
    try:
        rows = _get(STABLE_BASE + "/historical-price-eod/light",
                    {"symbol": symbol, "from": frm, "to": today.isoformat()}, api_key)
    except Exception as e:
        log.warning("benchmark: %s", e)
        return {}
    out = {}
    for r in rows if isinstance(rows, list) else []:
        d = str(r.get("date"))[:10]
        v = r.get("close", r.get("price"))
        if d and v:
            out[d] = float(v)
    return out


def fetch_index_constituents(index: str, api_key: str) -> list[str]:
    """Constituents of 'sp500', 'nasdaq', 'dowjones', or 'etf:TICKER'."""
    idx = index.lower()
    if idx.startswith("etf:"):
        sym = idx[4:].upper()
        data = _get(STABLE_BASE + "/etf/holdings", {"symbol": sym}, api_key)
        out = set()
        for d in data or []:
            a = d.get("asset") or d.get("symbol")
            if a:
                out.add(a)
        return sorted(out)
    if idx not in INDEX_PATHS:
        raise ValueError(f"Unknown index '{index}'. Use: {', '.join(INDEX_PATHS)} or etf:TICKER")
    data = _get(STABLE_BASE + INDEX_PATHS[idx], {}, api_key)
    return sorted({d.get("symbol") for d in (data or []) if d.get("symbol")})
