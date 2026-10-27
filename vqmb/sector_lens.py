"""
VQMB - Sector Lens.

A top-down read of what the market is rewarding, sector by sector, and
long/short pair candidates WITHIN each sector. It reads only ranks and grades
the model already computes - composite and pillar ranks, group strength, A/D,
Safety, Darvas and the flags. Nothing is re-estimated, and a missing value
stays missing: it drops out of a median and fails an eligibility test.

A "sector" is the model group (§14). Every threshold lives in config (LENS_*);
profile weights come from config.PROFILES.

    sector_lens(scores_df, profile)  ->  (sectors_df, pairs_df, headline)

Pairs are candidates for a PM to read on the card. They are not trades and
not a basket.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as C
from . import ranking as R

PILLARS = [("V", "v_rank_u", "Value"), ("Q", "q_rank_u", "Quality"),
           ("B", "b_rank_u", "Biz Mom"), ("P", "p_rank_u", "Price Mom")]
NOUN = {"B": "business-momentum", "P": "tape", "Q": "quality", "V": "value"}
HEADLINE_ORDER = ["B", "P", "Q", "V"]
# flags counted per sector: model flag -> the chip code shown on screen
COUNTED_FLAGS = [("CONFLICT", "MIX"), ("HYGIENE", "HYG"), ("DISCRETE", "JMP"),
                 ("GATE", "GATE"), ("SHORT_HISTORY", "NEW")]
REGIME_TEXT = {
    "CONFIRMED": "Fundamentals and tape agree",
    "TAPE LAGS": "Business improving, price not yet",
    "TAPE LEADS": "Price ahead of the business",
    "BROKEN": "Business and tape both weak",
    "MIXED": "No clear pillar signal",
}
PLAYBOOK = {
    "CONFIRMED": "Business and price agree. Own the leaders; pair only where dispersion allows.",
    "TAPE LAGS": "Business improving, price not yet. The early hunting ground: look for the "
                 "first long to break out.",
    "TAPE LEADS": "Price ahead of the business. Crowding risk: shorts need the business to "
                  "confirm before you press.",
    "BROKEN": "Both weak. Default underweight; a long needs a clean name with its own tape.",
}
DISPERSION_TEXT = {"WIDE": "Stock-picker's sector: pairs pay.",
                   "MODERATE": "Some room to pair.",
                   "TIGHT": "Names move together: trade the sector."}
NO_PAIR = {
    "no_long": "No confirmed long. Every strong name fails the tape or safety check.",
    "no_short": "No qualifying short. The sector is moving as a block, so own it or skip "
                "it rather than pair it.",
    "narrow": f"Spread too narrow. No long and short sit {C.LENS_MIN_SPREAD:g}+ rank points apart.",
}


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _med(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.median()) if len(s) else None


def _pct(mask: pd.Series, of: pd.Series) -> float | None:
    """Share of names (those with data) meeting `mask`, in percent."""
    n = int(of.sum())
    return round(100.0 * int((mask & of).sum()) / n, 1) if n else None


def _bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() == "true"
    try:
        return bool(v) and not (isinstance(v, float) and math.isnan(v))
    except (TypeError, ValueError):
        return False


def _prepare(df: pd.DataFrame, profile: str) -> pd.DataFrame:
    """One tidy frame: rank for the profile, pillar ranks, grades, flags."""
    rank_col = f"composite_rank_{profile}"
    if rank_col not in df.columns:
        raise KeyError(f"scores frame has no {rank_col}")
    x = pd.DataFrame(index=df.index)
    x["sym"] = df["symbol"].astype(str)
    x["name"] = df.get("company", pd.Series("", index=df.index)).fillna("").astype(str)
    x["sector"] = df["group"]
    x["rnk"] = pd.to_numeric(df[rank_col], errors="coerce")
    for k, col, _ in PILLARS:
        x[k] = pd.to_numeric(df.get(col), errors="coerce")
    for col, out in (("ad", "ad"), ("safety", "sf"), ("darvas", "dv"), ("darvas_stack", "st")):
        x[out] = df.get(col, pd.Series("", index=df.index)).fillna("").astype(str)
    gated = df.get("gated", pd.Series(False, index=df.index)).map(_bool)
    flags = []
    for i, row in df.iterrows():
        f = [k for k in C.FLAG_SEVERITY if k != "FRAGILE" and _bool(row.get(f"flag_{k}", False))]
        if gated.loc[i] and "GATE" not in f:
            f.insert(0, "GATE")
        flags.append(f)
    x["flags"] = flags
    # group strength for THIS profile: median member composite, re-ranked (§14)
    if x["sector"].notna().any():
        g = R.group_layer(df.assign(**{rank_col: x["rnk"]}), profile)["GRP"]
        x["grp"] = x["sector"].map(g)
    else:
        x["grp"] = np.nan
    return x[x["sector"].notna() & (x["sector"].astype(str) != "")]


def _regime(b, p) -> str:
    if b is None or p is None:
        return "MIXED"
    hi, lo = C.LENS_REGIME_HIGH, C.LENS_REGIME_LOW
    if b >= hi and p >= hi:
        return "CONFIRMED"
    if b >= hi and p < lo:
        return "TAPE LAGS"
    if b < lo and p >= hi:
        return "TAPE LEADS"
    if b < lo and p < lo:
        return "BROKEN"
    return "MIXED"


def _dispersion(iqr):
    if iqr is None:
        return None
    if iqr >= C.LENS_DISPERSION_WIDE:
        return "WIDE"
    if iqr < C.LENS_DISPERSION_TIGHT:
        return "TIGHT"
    return "MODERATE"


def _signed(v: float) -> str:
    r = int(round(v))
    return "±0" if r == 0 else (f"+{r}" if r > 0 else f"−{abs(r)}")


def _sectors(x: pd.DataFrame) -> pd.DataFrame:
    uni = {k: _med(x[k]) for k, _, _ in PILLARS}
    rows = []
    for sec, g in x.groupby("sector", sort=False):
        have = g["rnk"].notna()
        n = len(g)
        r = {"sector": str(sec), "n": n, "rnk": _med(g["rnk"])}
        for k, _, _ in PILLARS:
            m = _med(g[k])
            r[k] = m
            r[f"d{k}"] = None if m is None or uni[k] is None else m - uni[k]
        r["grp"] = _med(g["grp"])
        r["breadth"] = _pct(g["rnk"] >= C.LENS_BREADTH_RANK, have)
        r["laggards"] = _pct(g["rnk"] <= C.LENS_LAGGARD_RANK, have)
        ranks = g["rnk"].dropna()
        iqr = (float(ranks.quantile(0.75) - ranks.quantile(0.25))
               if len(ranks) >= C.LENS_DISPERSION_MIN_N else None)
        r["iqr"], r["dispersion"] = iqr, _dispersion(iqr)
        ad_have = g["ad"].isin(list("ABCDE"))
        r["ad_ab"] = _pct(g["ad"].isin(["A", "B"]), ad_have)
        r["ad_de"] = _pct(g["ad"].isin(["D", "E"]), ad_have)
        sf_have = g["sf"].isin(list("ABCDE"))
        r["sf_de"] = _pct(g["sf"].isin(["D", "E"]), sf_have)
        r["box"] = int((g["dv"] == "BOX").sum())
        for flag, code in COUNTED_FLAGS:
            r[f"f_{code}"] = int(g["flags"].map(lambda f: flag in f).sum())
        r["regime"] = _regime(r["B"], r["P"])
        r["thin"] = n < C.LENS_MIN_NAMES
        r["trait"] = _trait(r)
        rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["rnk", "n"], ascending=[False, False], na_position="last")
    return out.reset_index(drop=True)


def _trait(r: dict) -> str:
    """One sentence: strongest and weakest pillar vs the universe, breadth,
    then the dispersion verdict."""
    names = {k: lab for k, _, lab in PILLARS}
    d = {k: r[f"d{k}"] for k, _, _ in PILLARS if r[f"d{k}"] is not None}
    bits = []
    if d:
        hi = max(d, key=d.get)
        lo = min(d, key=d.get)
        bits.append(f"Strongest on {names[hi]} ({_signed(d[hi])} vs universe), "
                    f"weakest on {names[lo]} ({_signed(d[lo])}).")
    if r["breadth"] is not None:
        bits.append(f"{r['breadth']:.0f}% of names rank {C.LENS_BREADTH_RANK:g}+.")
    if r["dispersion"]:
        bits.append(DISPERSION_TEXT[r["dispersion"]])
    return " ".join(bits)


# ---------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------
def _long_ok(r) -> bool:
    return (r["rnk"] is not None and r["rnk"] >= C.LENS_LONG_MIN_RANK
            and r["P"] is not None and r["P"] >= C.LENS_LONG_MIN_P
            and r["ad"] not in C.LENS_LONG_BAD_AD and r["ad"] != ""
            and r["sf"] not in C.LENS_LONG_BAD_SAFETY and r["sf"] != ""
            and not any(f in r["flags"] for f in C.LENS_LONG_BAD_FLAGS))


def _short_ok(r, bar: float) -> bool:
    weak = ((r["B"] is not None and r["B"] <= C.LENS_SHORT_MAX_B)
            or (r["P"] is not None and r["P"] <= C.LENS_SHORT_MAX_P))
    return (r["rnk"] is not None and r["rnk"] <= bar and weak
            and not any(f in r["flags"] for f in C.LENS_SHORT_BAD_FLAGS))


def short_bar(ranks: pd.Series) -> float:
    """Short rank ceiling: max(40, min(50, the sector's 33rd percentile))."""
    s = pd.to_numeric(ranks, errors="coerce").dropna()
    p33 = float(np.percentile(s, C.LENS_SHORT_RANK_PCTL)) if len(s) else C.LENS_SHORT_RANK_FLOOR
    return max(C.LENS_SHORT_RANK_FLOOR, min(C.LENS_SHORT_RANK_CAP, p33))


def decompose(lg: dict, sh: dict, weights: dict) -> dict:
    """Contribution of each pillar = weight x (pillar_long - pillar_short); its
    share is of the total ABSOLUTE contribution, so a pillar working against
    the pair shows a negative share. A missing pillar contributes nothing."""
    out = {}
    for k, _, _ in PILLARS:
        a, b = lg.get(k), sh.get(k)
        dlt = None if a is None or b is None else a - b
        out[k] = {"delta": dlt, "contrib": 0.0 if dlt is None else weights[k] / 100.0 * dlt}
    tot = sum(abs(v["contrib"]) for v in out.values())
    for v in out.values():
        v["share"] = v["contrib"] / tot if tot else None
    return out


def pair_tags(lg: dict, sh: dict, parts: dict) -> list[tuple[str, str]]:
    """(tone, text): ok = a check passed, warn = read the card."""
    tags = []
    dv = parts["V"]["delta"]
    if dv is not None:
        if abs(dv) <= C.LENS_TAG_VALUE_CLEAN:
            tags.append(("ok", "Style-clean on value"))
        elif dv >= C.LENS_TAG_VALUE_BIG:
            tags.append(("ok", "Long is also cheaper"))
        elif dv <= -C.LENS_TAG_VALUE_BIG:
            tags.append(("warn", f"Paying up: long is {abs(round(dv))} pts richer"))
    bs, ps = parts["B"]["share"], parts["P"]["share"]
    if bs is not None and ps is not None and bs < C.LENS_TAG_TAPE_B_SHARE and ps > C.LENS_TAG_TAPE_P_SHARE:
        tags.append(("warn", "Tape-only spread: business does not confirm"))
    if sh["P"] is not None and sh["P"] >= C.LENS_TAG_SHORT_STRONG_P:
        tags.append(("warn", f"Short still has a strong tape (P {round(sh['P'])}): wait for a break"))
    if (sh["V"] is not None and sh["Q"] is not None
            and sh["V"] >= C.LENS_TAG_SQUEEZE_V and sh["Q"] >= C.LENS_TAG_SQUEEZE_Q):
        tags.append(("warn", "Short is cheap and high quality: squeeze risk"))
    if sh["ad"] in ("D", "E"):
        tags.append(("ok", f"Short under distribution (A/D {sh['ad']})"))
    elif sh["ad"] in ("A", "B"):
        tags.append(("warn", f"Short under accumulation (A/D {sh['ad']}): buyers present"))
    if lg["ad"] == "A":
        tags.append(("ok", "Long under accumulation"))
    elif lg["ad"] == "D":
        tags.append(("warn", "Long under distribution (A/D D): size down"))
    if "CONFLICT" in lg["flags"]:
        tags.append(("warn", "Long carries MIX: read the card"))
    return tags


def _rec(row) -> dict:
    return {"sym": row["sym"], "name": row["name"], "rnk": _num(row["rnk"]),
            **{k: _num(row[k]) for k, _, _ in PILLARS},
            "ad": row["ad"], "sf": row["sf"], "dv": row["dv"], "st": row["st"],
            "flags": list(row["flags"])}


def _pairs(x: pd.DataFrame, sectors: pd.DataFrame, weights: dict):
    pairs, why = [], {}
    names = {k: lab for k, _, lab in PILLARS}
    for _, s in sectors.iterrows():
        if s["thin"]:
            continue
        g = x[x["sector"].astype(str) == s["sector"]]
        recs = [_rec(r) for _, r in g.iterrows()]
        bar = short_bar(g["rnk"])
        longs = [r for r in recs if _long_ok(r)]
        shorts = [r for r in recs if _short_ok(r, bar)]
        cands = []
        for lg in longs:
            for sh in shorts:
                if lg["sym"] == sh["sym"]:
                    continue
                spread = lg["rnk"] - sh["rnk"]
                if spread < C.LENS_MIN_SPREAD:
                    continue
                dv = (lg["V"] - sh["V"]) if lg["V"] is not None and sh["V"] is not None else 0.0
                score = spread - C.LENS_VALUE_PENALTY * max(0.0, -dv - C.LENS_VALUE_PENALTY_FREE)
                cands.append((score, spread, lg, sh))
        cands.sort(key=lambda c: (-c[0], -c[1], c[2]["sym"], c[3]["sym"]))
        used, picked = set(), 0
        for score, spread, lg, sh in cands:
            if picked >= C.LENS_MAX_PAIRS:
                break
            if lg["sym"] in used or sh["sym"] in used:
                continue
            used |= {lg["sym"], sh["sym"]}
            picked += 1
            parts = decompose(lg, sh, weights)
            driver = max(parts, key=lambda k: parts[k]["contrib"])
            pairs.append({"sector": s["sector"], "long": lg, "short": sh,
                          "spread": spread, "score": score, "driver": names[driver],
                          "parts": parts, "tags": pair_tags(lg, sh, parts),
                          "short_bar": bar})
        if not picked:
            why[s["sector"]] = ("no_long" if not longs else "no_short" if not shorts else "narrow")
    return pairs, why


# ---------------------------------------------------------------------------
# the top-down read
# ---------------------------------------------------------------------------
def _headline(sectors: pd.DataFrame) -> dict:
    names = {k: lab for k, _, lab in PILLARS}
    pool = sectors[~sectors["thin"] & sectors["rnk"].notna()].sort_values("rnk", ascending=False)
    k = min(C.LENS_LEADERS, len(pool) // 2)
    regimes = pool["regime"].value_counts().to_dict() if len(pool) else {}
    if k < 1:
        return {"text": "Too few sectors with four or more names to compare leaders and laggards.",
                "leaders": [], "laggards": [], "gaps": {}, "regimes": regimes}
    lead, lag = pool.head(k), pool.tail(k)
    gaps = {}
    for p in HEADLINE_ORDER:
        a, b = lead[p].dropna(), lag[p].dropna()
        gaps[p] = float(a.mean() - b.mean()) if len(a) and len(b) else None
    clear, flat = C.LENS_PILLAR_GAP_CLEAR, C.LENS_PILLAR_GAP_FLAT
    pos = [p for p in HEADLINE_ORDER if gaps[p] is not None and gaps[p] >= clear]
    neg = [p for p in HEADLINE_ORDER if gaps[p] is not None and gaps[p] <= -clear]
    small = [p for p in HEADLINE_ORDER if gaps[p] is not None and abs(gaps[p]) < flat]
    on = lambda ps: " and ".join(f"{_signed(gaps[p])} on {names[p]}" for p in ps)

    if pos:
        lead_s = "Leadership is a " + " and ".join(NOUN[p] for p in pos) + " story"
        if "V" in neg:
            lead_s += ", and the market is paying up for it."
        elif gaps.get("V") is not None and gaps["V"] >= clear and "V" not in pos:
            lead_s += ", and it comes cheap."
        else:
            lead_s += "."
    else:
        lead_s = "No pillar clearly separates the leading sectors from the lagging ones."
    words = {1: "strongest sector", 2: "two strongest sectors", 3: "three strongest sectors"}
    weak = {1: "weakest", 2: "two weakest", 3: "three weakest"}
    body = ""
    if pos or neg:
        body = f" The {words.get(k, f'{k} strongest sectors')} beat the {weak.get(k, f'{k} weakest')}"
        body += f" by {on(pos)}" if pos else ""
        if neg:
            body += (", while sitting " if pos else " yet sit ") + on(neg)
        body += "."
    if small:
        body += " " + " and ".join(names[p] for p in small) + (
            " barely separates them" if len(small) == 1 else " barely separate them") + \
            " (" + ", ".join(_signed(gaps[p]) for p in small) + ")."
    return {"text": (lead_s + body).strip(), "leaders": lead["sector"].tolist(),
            "laggards": lag["sector"].tolist(), "gaps": gaps, "regimes": regimes}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def sector_lens(scores_df: pd.DataFrame, profile: str = C.BASE_PROFILE):
    """(sectors_df, pairs_df, headline) for one profile.

    scores_df is the full scored frame (engine.score); the weights are that
    profile's from config.PROFILES."""
    weights = C.PROFILES[profile]
    x = _prepare(scores_df, profile)
    sectors = _sectors(x)
    if not len(sectors):
        return sectors, pd.DataFrame(), {"text": "", "leaders": [], "laggards": [],
                                         "gaps": {}, "regimes": {}}
    pairs, why = _pairs(x, sectors, weights)
    sectors["no_pair"] = sectors["sector"].map(why)
    head = _headline(sectors)
    return sectors, _pairs_frame(pairs, profile), head | {"_pairs": pairs}


def _pairs_frame(pairs: list, profile: str) -> pd.DataFrame:
    rows = []
    for p in pairs:
        lg, sh = p["long"], p["short"]
        r = {"profile": profile, "sector": p["sector"], "long": lg["sym"], "short": sh["sym"],
             "long_rank": lg["rnk"], "short_rank": sh["rnk"], "spread": p["spread"],
             "score": round(p["score"], 2), "driver": p["driver"]}
        for k, _, lab in PILLARS:
            r[f"long_{k}"], r[f"short_{k}"] = lg[k], sh[k]
            r[f"delta_{k}"] = p["parts"][k]["delta"]
            r[f"contrib_{k}"] = round(p["parts"][k]["contrib"], 3)
            sh_ = p["parts"][k]["share"]
            r[f"share_{k}"] = None if sh_ is None else round(sh_, 4)
        r.update(long_ad=lg["ad"], long_safety=lg["sf"], short_ad=sh["ad"], short_safety=sh["sf"],
                 long_flags=" ".join(lg["flags"]), short_flags=" ".join(sh["flags"]),
                 tags=" | ".join(t for _, t in p["tags"]))
        rows.append(r)
    return pd.DataFrame(rows)


def _clean(v):
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    return v


def lens_attrs(df: pd.DataFrame) -> dict:
    """Sector Lens for every profile, as plain records for df.attrs:
    {profile: {"sectors": [...], "pairs": [...], "headline": {...}}}.
    pd.DataFrame(df.attrs["sector_lens"]["TRADER"]["sectors"]) gives the table."""
    out = {}
    for prof in C.PROFILES:
        sectors, pairs, head = sector_lens(df, prof)
        head = {k: _clean(v) for k, v in head.items() if not k.startswith("_")}
        if len(sectors):
            sectors = sectors.drop(columns=[c for c in sectors.columns if c.startswith("_")])
        out[prof] = {
            "sectors": [{k: _clean(v) for k, v in r.items()} for r in sectors.to_dict(orient="records")],
            "pairs": [{k: _clean(v) for k, v in r.items()} for r in pairs.to_dict(orient="records")],
            "headline": head,
        }
    return out
