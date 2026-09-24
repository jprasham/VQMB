"""
VQMB — Darvas behaviour states and the A/D grade (handbook §15, §16).

Both are unscored. They classify and inform; they never enter a rank. The
Darvas floor gives the trader a mechanical stop, which is the whole point of
pairing it with the fundamental F-stop on the card.

The asymmetry between the two grades is deliberate and documented so nobody
"fixes" it later: Safety is absolute (a fortress is a fortress regardless of
peers) while A/D is relative (fixed volume-ratio thresholds would grade the
market, not the stock).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as C

NA = float("nan")


def darvas_state(dates, closes, highs, lows, volumes) -> dict:
    """Classify one name's current box state.

    A box forms when a high goes unexceeded for three consecutive sessions
    (the ceiling) and a subsequent low holds for three (the floor). Breakouts
    need volume confirmation - without it there is no state change at all,
    which is what stops a drifting tape from manufacturing signals.

    Returns state, box top/floor, days in box, tightness, and the stack count.
    """
    out = {"darvas": "DRIFT", "darvas_stack": "", "box_top": NA, "box_floor": NA,
           "days_in_box": 0, "tightness": NA, "stack": 0}
    if not closes or len(closes) < 60:
        return out

    n = len(closes)
    price = closes[-1]
    hi252 = max([h for h in highs[-252:] if math.isfinite(h)] or [NA])

    # Eligibility (§15): a name more than 5% off its 52-week high is DRIFT,
    # whatever its recent shape - the box only means something near the highs.
    if not math.isfinite(hi252) or hi252 <= 0 or price / hi252 - 1.0 < -C.DARVAS_ELIGIBLE_PCT:
        return out

    k = C.DARVAS_CONFIRM_SESSIONS
    boxes = []          # (ceiling, floor, start_index)
    i = k
    while i < n - k:
        ceiling = highs[i]
        if not math.isfinite(ceiling):
            i += 1
            continue
        # ceiling must stand unexceeded for k sessions
        if all(math.isfinite(highs[j]) and highs[j] <= ceiling for j in range(i + 1, i + 1 + k)):
            floor, fi = NA, None
            for j in range(i + 1, min(n - k, i + 40)):
                cand = lows[j]
                if not math.isfinite(cand):
                    continue
                if all(math.isfinite(lows[m]) and lows[m] >= cand
                       for m in range(j + 1, j + 1 + k)):
                    floor, fi = cand, j
                    break
            if fi is not None and math.isfinite(floor) and ceiling > 0:
                height = (ceiling - floor) / ceiling
                if height <= C.DARVAS_MAX_BOX_HEIGHT:
                    boxes.append((ceiling, floor, fi))
                    i = fi + k
                    continue
        i += 1

    if not boxes:
        return out
    # a box older than the drift window no longer describes the tape
    ceiling, floor, start = boxes[-1]
    if n - 1 - start > C.DARVAS_DRIFT_SESSIONS:
        return out

    out["box_top"], out["box_floor"] = ceiling, floor
    out["days_in_box"] = n - 1 - start
    out["tightness"] = (ceiling - floor) / ceiling if ceiling else NA

    # consecutive boxes with rising floors
    stack = 1
    for a, b in zip(reversed(boxes[:-1]), reversed(boxes[1:])):
        if math.isfinite(a[1]) and math.isfinite(b[1]) and b[1] > a[1]:
            stack += 1
        else:
            break
    out["stack"] = stack

    vol50 = np.nanmean([v for v in volumes[-50:] if math.isfinite(v)]) \
        if len(volumes) >= 50 else NA
    last_vol = volumes[-1] if volumes else NA

    # BRK UP and STACK are separate ideas in the handbook, not alternatives: a
    # name can be breaking out AND standing on a stack of rising floors. Making
    # them exclusive hid every stacked breakout behind the stack label, which is
    # exactly the setup the techno-fundamental list is looking for.
    if price > ceiling:
        # no volume, no state change - the handbook is explicit
        if math.isfinite(vol50) and math.isfinite(last_vol) \
                and last_vol >= C.DARVAS_BREAKOUT_VOLUME * vol50:
            out["darvas"] = "BRK UP"
        else:
            out["darvas"] = "BOX"      # above the ceiling but unconfirmed
    elif price < floor:
        out["darvas"] = "BRK DOWN"
    else:
        out["darvas"] = "BOX"
    if stack > 1:
        out["darvas_stack"] = f"STACK x{stack}"
    return out


def ad_ratio(closes, volumes, sessions: int = C.AD_SESSIONS) -> float:
    """Up-volume divided by down-volume over the trailing window. Raw ratio;
    the letter grade is assigned by universe quintile, not a fixed cut."""
    if not closes or len(closes) < sessions + 2:
        return NA
    up = dn = 0.0
    for i in range(len(closes) - sessions, len(closes)):
        if i <= 0 or not math.isfinite(closes[i]) or not math.isfinite(closes[i - 1]):
            continue
        # row 18: a day with no volume figure is skipped, not counted as zero
        # volume - a zero would silently shrink whichever side it fell on
        if i >= len(volumes) or not math.isfinite(volumes[i]):
            continue
        v = volumes[i]
        if closes[i] > closes[i - 1]:
            up += v
        elif closes[i] < closes[i - 1]:
            dn += v
    if dn <= 0:
        return NA if up <= 0 else 99.0
    return up / dn


def ad_grade(ratios: pd.Series) -> pd.Series:
    """Universe quintiles A-E. Relative by design: a fixed threshold would
    grade the market rather than the stock, because accumulation is only
    meaningful against what everything else is doing."""
    v = pd.to_numeric(ratios, errors="coerce")
    if v.notna().sum() < 5:
        return pd.Series(pd.NA, index=ratios.index, dtype="object")
    q = v.rank(pct=True, na_option="keep")
    out = pd.Series(pd.NA, index=ratios.index, dtype="object")
    out[q > 0.8] = "A"
    out[(q > 0.6) & (q <= 0.8)] = "B"
    out[(q > 0.4) & (q <= 0.6)] = "C"
    out[(q > 0.2) & (q <= 0.4)] = "D"
    out[q <= 0.2] = "E"
    return out
