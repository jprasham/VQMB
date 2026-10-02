"""
VQMB — the rank chain (handbook §3, §9, §10, §11, §13, §16).

    metric raw values
      -> metric percentiles, computed twice: within universe (U) and sector (S)
      -> pillar raw: mean of available metric percentiles; Quality and Price
         multiply by a dampener
      -> pillar ranks: pillar raw re-percentiled -> RNK.U and RNK.S
      -> composite raw: profile-weighted sum of pillar RNK.U
      -> composite rank: re-percentiled within universe, per profile
      -> overlays: gates exclude, caps limit - OUTSIDE every average

Two rules carry most of the weight:

  Everything a PM sorts by is a rank. Raw values are kept for diagnostics and
  attribution, never as a sort key.

  Gates and caps sit outside all averages and can never be bought back by
  strength elsewhere. A dampener discounts; a gate excludes.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# percentiles
# ---------------------------------------------------------------------------
def percentile(s: pd.Series, direction: int = 1) -> pd.Series:
    """Rank position / (group size - 1) * 100, direction applied, ties averaged.

    Handbook §3. A one-name group is undefined rather than 0 or 100, so it
    returns NaN: with nobody to compare against there is no percentile.
    """
    v = pd.to_numeric(s, errors="coerce")
    if direction < 0:
        v = -v
    n = v.notna().sum()
    if n < 2:
        return pd.Series(np.nan, index=s.index, dtype="float64")
    r = v.rank(method="average", na_option="keep")
    return (r - 1) / (n - 1) * 100.0


def percentile_spliced(df: pd.DataFrame, col: str, direction: int,
                       fin_direction: int | None = None,
                       min_group: int = C.MIN_GROUP_SIZE) -> pd.Series:
    """Rank financials among financials and everyone else among everyone else,
    then splice the two into one column (§12).

    The variant metrics measure different quantities for the two groups, so a
    single universe-wide ranking would compare a bank's ROE against an
    industrial's ROIC. After splicing, U == S for the financial rows by
    construction - the card prints those with an asterisk.

    Too few financials to rank among themselves -> fall back to one universe
    ranking, because a percentile over eight names is noise.
    """
    fin = df.get("is_fin")
    fin = (fin.fillna(False).astype(bool) if fin is not None
           else pd.Series(False, index=df.index))
    if fin.sum() < min_group or (~fin).sum() < min_group:
        return percentile(df[col], direction)
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out[~fin] = percentile(df.loc[~fin, col], direction)
    out[fin] = percentile(df.loc[fin, col], fin_direction or direction)
    return out


def percentile_within(df: pd.DataFrame, col: str, group_col: str | None,
                      direction: int = 1, min_group: int = C.MIN_GROUP_SIZE) -> pd.Series:
    """Percentile within each group. Groups below min_group fall back to the
    whole universe, so a thin sector never produces a rank off three names."""
    if group_col is None or group_col not in df.columns:
        return percentile(df[col], direction)
    out = percentile(df[col], direction)          # universe-wide default
    for _, idx in df.groupby(group_col, dropna=True).groups.items():
        if len(idx) >= min_group:
            sub = percentile(df.loc[idx, col], direction)
            out.loc[idx] = sub
    return out


# ---------------------------------------------------------------------------
# blanks and reweighting (§13) - one law
# ---------------------------------------------------------------------------
def block_mean(pcts: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Mean of the percentiles in one sub-block.

    With config.METRIC_NEUTRAL_FILL_ON, a blank metric counts as the neutral
    percentile (50) and the mean runs over every metric in the sub-block. The
    text below describes the FLUX rule, used when the switch is off.

    A blank metric is removed and its siblings reweight equally among
    themselves. A sub-block never borrows weight across the divide - value
    among value, engine among engine, shield among shield. Since every input is
    already a 0-100 percentile, an equal-weight mean of whatever survives IS
    the reweighting rule; nothing further is needed.
    """
    if C.METRIC_NEUTRAL_FILL_ON:
        # a metric never computed for anyone is a column of blanks - neutral too
        return pcts.reindex(columns=cols).fillna(C.METRIC_NEUTRAL_FILL).mean(axis=1)
    have = [c for c in cols if c in pcts.columns]
    if not have:
        return pd.Series(np.nan, index=pcts.index, dtype="float64")
    return pcts[have].mean(axis=1, skipna=True)


# ---------------------------------------------------------------------------
# pillars
# ---------------------------------------------------------------------------
def build_pillars(raw: pd.DataFrame, pcts: pd.DataFrame) -> pd.DataFrame:
    """Pillar raw scores, with the two dampeners applied before re-ranking."""
    out = pd.DataFrame(index=raw.index)

    out["value_raw"] = block_mean(pcts, [k for k, _ in C.VALUE_METRICS])

    engine = block_mean(pcts, [k for k, _ in C.QUALITY_ENGINE])
    shield = block_mean(pcts, [k for k, _ in C.QUALITY_SHIELD])
    out["quality_engine"] = engine
    out["quality_shield"] = shield
    # Multiplicative on purpose: a fragile balance sheet does not subtract a
    # fixed amount from quality, it discounts all of it. And protection with no
    # engine earns nothing - engine 0 stays 0 whatever the shield.
    # row 10: a blank shield is neutral, not best case
    out["quality_damp"] = C.SHIELD_FLOOR + C.SHIELD_SPAN * shield.fillna(C.DAMPENER_NEUTRAL_FILL) / 100.0
    out["quality_raw"] = engine * out["quality_damp"]

    out["biz_raw"] = block_mean(pcts, [k for k, _ in C.BIZ_METRICS])

    strength = block_mean(pcts, [k for k, _ in C.PRICE_STRENGTH])
    cred = block_mean(pcts, [k for k, _ in C.PRICE_CREDIBILITY])
    out["price_strength"] = strength
    out["price_credibility"] = cred
    out["price_damp"] = C.CRED_FLOOR + C.CRED_SPAN * cred.fillna(C.DAMPENER_NEUTRAL_FILL) / 100.0
    out["price_raw"] = strength * out["price_damp"]
    return out


def rank_pillars(df: pd.DataFrame, sector_col: str | None = "sector") -> pd.DataFrame:
    """Re-percentile each pillar raw into RNK.U and RNK.S - the four sortable
    sub-models."""
    out = pd.DataFrame(index=df.index)
    for pillar, col in (("V", "value_raw"), ("Q", "quality_raw"),
                        ("B", "biz_raw"), ("P", "price_raw")):
        out[f"{pillar.lower()}_rank_u"] = percentile(df[col], +1)
        out[f"{pillar.lower()}_rank_s"] = percentile_within(df, col, sector_col, +1)
    return out


# ---------------------------------------------------------------------------
# composites
# ---------------------------------------------------------------------------
def build_composites(ranks: pd.DataFrame) -> pd.DataFrame:
    """Profile-weighted sum of the four pillar RNK.U values, then re-percentiled
    within the universe, per profile. All three profiles are computed for every
    name every night; the screen shows the active one."""
    out = pd.DataFrame(index=ranks.index)
    cols = {"V": "v_rank_u", "Q": "q_rank_u", "B": "b_rank_u", "P": "p_rank_u"}
    for name, w in C.PROFILES.items():
        num = sum(w[p] * ranks[cols[p]].fillna(0.0) for p in w)
        den = sum(w[p] * ranks[cols[p]].notna() for p in w)
        # a pillar with no computable metric is dropped and the rest reweight,
        # matching the blank law one level up
        raw = num / den.replace(0, np.nan) * sum(w.values()) / 100.0
        raw = raw * 100.0 / sum(w.values())
        # row 30: one surviving pillar must not carry the whole rank
        present = sum(ranks[cols[p]].notna().astype(int) for p in w)
        need = int(np.ceil(C.COMPOSITE_MIN_PILLAR_SHARE * len(w)))
        raw = raw.where(present >= need)
        out[f"composite_raw_{name}"] = raw
        out[f"composite_rank_{name}"] = percentile(raw, +1)
    cnt = sum(ranks[c].notna().astype(int) for c in cols.values())
    out["pillars_present"] = cnt
    out["thin"] = cnt < int(np.ceil(C.COMPOSITE_MIN_PILLAR_SHARE * len(cols)))
    return out


# ---------------------------------------------------------------------------
# overlays: gates, conviction, flags (§10, §11, §16)
# ---------------------------------------------------------------------------
def apply_gates(df: pd.DataFrame) -> pd.Series:
    """True where the name is GATED. Gates exclude from green and sit at the
    bottom of the screen with their cause; they are never hidden."""
    gated = pd.Series(False, index=df.index)
    cause = pd.Series("", index=df.index, dtype="object")

    # The leverage gate is for operating companies ONLY. A lender's "net debt"
    # is its deposits and borrowings - the raw material of the business, not a
    # financing choice - so ND/EBIT runs into the hundreds and would gate every
    # bank in the index. That is precisely why the handbook gives financials a
    # separate capital gate (§11) rather than one rule for everyone.
    is_fin = df.get("is_fin")
    is_fin = (is_fin.fillna(False).astype(bool) if is_fin is not None
              else pd.Series(False, index=df.index))
    operating = ~is_fin

    nd_ebit = df.get("nd_ebit")
    if nd_ebit is not None:
        hit = (pd.to_numeric(nd_ebit, errors="coerce") > C.LEVERAGE_GATE_ND_EBIT) & operating
        gated |= hit.fillna(False)
        cause[hit.fillna(False)] = f"ND/EBIT > {C.LEVERAGE_GATE_ND_EBIT:g}x"

    neg = df.get("ebit_negative_with_net_debt")
    if neg is not None:
        hit = neg.fillna(False).astype(bool) & operating
        gated |= hit
        cause[hit & (cause == "")] = "EBIT <= 0 with net debt"

    cap = df.get("capital_ratio")
    if cap is not None and "fin_kind" in df.columns:
        bank = df["fin_kind"].isin(["bank", "nbfc"])
        ins = df["fin_kind"] == "insurer"
        hit = ((bank & (pd.to_numeric(cap, errors="coerce") < C.BANK_CAPITAL_GATE))
               | (ins & (pd.to_numeric(cap, errors="coerce") < C.INSURER_CAPITAL_GATE)))
        gated |= hit.fillna(False)
        cause[hit.fillna(False) & (cause == "")] = "capital below floor"

    # row 11: a gate raised by the leverage metric (Q4) must never appear with
    # an empty cause; the metric and the gate share one rule for missing EBIT
    lev = df.get("leverage_gate")
    if lev is not None:
        hit = lev.fillna(False).astype(bool)
        gated |= hit
        cause[hit & (cause == "")] = "leverage gate (Q4)"

    df["gate_cause"] = cause
    return gated


def conviction(df: pd.DataFrame, profile: str) -> pd.DataFrame:
    """Green and the triple star (§10).

    Green needs BOTH a top-5% rank and a raw composite a set distance above the
    universe median - the dispersion floor. In a flat market where everything
    scores alike, nothing is green, and that variation is information rather
    than a fault.
    """
    out = pd.DataFrame(index=df.index)
    rank = df[f"composite_rank_{profile}"]
    raw = df[f"composite_raw_{profile}"]
    median = raw.median(skipna=True)
    out["green"] = ((rank >= 100 - C.GREEN_TOP_PCT)
                    & (raw >= median + C.GREEN_DISPERSION_FLOOR)
                    & ~df.get("gated", pd.Series(False, index=df.index)))
    top_decile = [df[f"composite_rank_{p}"] >= 100 - C.TRIPLE_TOP_PCT for p in C.PROFILES]
    out["triple"] = np.logical_and.reduce(top_decile) & ~df.get(
        "gated", pd.Series(False, index=df.index))
    return out


def safety_grade(shield: pd.Series, drawdown_pct: pd.Series,
                 gated: pd.Series) -> pd.DataFrame:
    """Absolute by design - a fortress is a fortress regardless of peers, so
    fixed bands rather than universe quintiles (contrast A/D, which is
    relative). Any gate forces E."""
    # row 10: the same neutral fill the pillar dampener uses, not worst case
    raw = (C.SAFETY_SHIELD_W * shield.fillna(C.DAMPENER_NEUTRAL_FILL)
           + C.SAFETY_DRAWDOWN_W * drawdown_pct.fillna(0.0))

    def band(x):
        for floor, letter in C.SAFETY_BANDS:
            if x >= floor:
                return letter
        return "E"

    grade = raw.map(band)
    grade[gated.fillna(False)] = "E"
    return pd.DataFrame({"safety_raw": raw, "safety": grade})


def compute_flags(df: pd.DataFrame, previous: pd.DataFrame | None = None) -> pd.Series:
    """Flags with hysteresis (§16).

    A flag that is already on stays on until the looser OFF threshold is
    crossed, so a name sitting near a boundary does not blink night after
    night. `previous` is yesterday's flag set; without it every flag simply
    uses its ON threshold.
    """
    def was_on(flag, idx):
        if previous is None or flag not in previous.columns:
            return pd.Series(False, index=idx)
        return previous[flag].reindex(idx).fillna(False).astype(bool)

    flags = pd.DataFrame(index=df.index)
    flags["GATE"] = df.get("gated", pd.Series(False, index=df.index)).fillna(False)

    hyg = pd.to_numeric(df.get("q6_hygiene_pct"), errors="coerce")
    on = hyg < C.HYGIENE_ON
    stay = was_on("HYGIENE", df.index) & (hyg < C.HYGIENE_OFF)
    flags["HYGIENE"] = (on | stay).fillna(False)

    st = pd.to_numeric(df.get("price_strength"), errors="coerce")
    cr = pd.to_numeric(df.get("price_credibility"), errors="coerce")
    on = (st >= C.DISCRETE_STRENGTH_ON) & (cr <= C.DISCRETE_CRED_ON)
    stay = was_on("DISCRETE", df.index) & ~((cr > C.DISCRETE_CRED_OFF)
                                            | (st < C.DISCRETE_STRENGTH_OFF))
    flags["DISCRETE"] = (on | stay).fillna(False)

    pill = df[["v_rank_u", "q_rank_u", "b_rank_u", "p_rank_u"]]
    spread = pill.max(axis=1) - pill.min(axis=1)
    on = spread >= C.CONFLICT_SPREAD_ON
    stay = was_on("CONFLICT", df.index) & (spread >= C.CONFLICT_SPREAD_OFF)
    flags["CONFLICT"] = (on | stay).fillna(False)
    flags["pillar_spread"] = spread

    B = lambda c: df.get(c, pd.Series(False, index=df.index)).fillna(False).astype(bool)
    for col, flag in (("short_history", "SHORT_HISTORY"), ("v3_loss", "LOSS"),
                      ("is_fin", "FIN")):
        flags[flag] = B(col)
    # the CALC chip says "model-built"; a blank CALC-NA yield must not carry it
    flags["CALC"] = B("v3_calc") & ~B("v3_calc_na")
    flags["GPA"] = B("q1_used_gpa_fallback") & ~B("is_fin")
    flags["FX"] = B("fx_mismatch")
    flags["STALE"] = B("stale")
    flags["THIN"] = B("thin")
    flags["PXCHK"] = B("px_mismatch")
    flags["WKCHK"] = B("workings_mismatch")
    # gross profit set aside as unusable - only meaningful for operating
    # companies, since financials never use gross profit
    flags["NOGP"] = df.get("gp_set_aside", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    flags["FRAGILE"] = False        # country fragility: specced, dormant until
                                    # a multi-market universe exists (§11)
    return flags


def screen_flags(flags: pd.DataFrame) -> pd.Series:
    """At most two flags on the screen row, in severity order. The card shows
    the full set - nothing is hidden, only deferred."""
    def pick(row):
        on = [f for f in C.FLAG_SEVERITY if row.get(f, False)]
        return ", ".join(on[:C.SCREEN_FLAG_LIMIT])
    return flags.apply(pick, axis=1)


# ---------------------------------------------------------------------------
# group layer (§14) - unscored
# ---------------------------------------------------------------------------
def group_layer(df: pd.DataFrame, profile: str, group_col: str = "group") -> pd.DataFrame:
    """Group strength: the median member composite RNK.U, re-ranked.

    Median rather than mean because groups run 15-40 names and one extreme
    member should not move the group. And because U percentiles are uniform
    across the universe, a group median of 71 reads directly as 'the typical
    member sits at the 71st percentile of the market'.

    Uses RNK.U, never RNK.S: sector ranks are uniform within a group by
    construction and would carry zero group information.
    """
    col = f"composite_rank_{profile}"
    g = df.groupby(group_col, dropna=True)
    tab = pd.DataFrame({
        "N": g[col].size(),
        "MED_CMP": g[col].median(),
        "MED_P": g["p_rank_u"].median(),
        "IQR": g[col].quantile(0.75) - g[col].quantile(0.25),
    })
    tab["GRP"] = percentile(tab["MED_CMP"], +1)
    return tab.sort_values("GRP", ascending=False)


def checklist(df: pd.DataFrame) -> pd.DataFrame:
    """Ten items, PASS / NEUTRAL (within 10 points) / FAIL, pinned to the
    TRADER profile for every user by design. No data -> NEUTRAL."""
    out = pd.DataFrame(index=df.index)
    passes = pd.Series(0, index=df.index)
    for label, col, bar in C.CHECKLIST:
        if col is None:
            out[label] = "NEUTRAL"          # B4 removed -> revisions item is permanently neutral
            continue
        if col not in df.columns:
            # row 4: fail loudly. Reading a wrong column name as "no data" is
            # what hid the RNK and GRP items reading NEUTRAL for every name.
            raise KeyError(f"checklist item {label!r} reads column {col!r}, "
                           "which is not in the scored frame")
        v = pd.to_numeric(df[col], errors="coerce")
        res = pd.Series("NEUTRAL", index=df.index, dtype="object")
        res[v >= bar] = "PASS"
        res[v < bar - C.CHECKLIST_NEUTRAL_BAND] = "FAIL"
        out[label] = res.where(v.notna(), "NEUTRAL")
        passes += (out[label] == "PASS").astype(int)
    out["pass_count"] = passes
    return out
