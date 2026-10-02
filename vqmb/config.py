"""
VQMB — frozen configuration.

Every model constant lives here: profile weights, dampener ranges, gate
thresholds, flag hysteresis, metric structure and the data-quality switches.
None of it is a runtime setting - the library exposes no knobs that change the
model.

Methodology decisions recorded here so they are visible rather than buried in
code:

  B4 (revisions composite) REMOVED. It needs a 3-month history of daily
      consensus snapshots that FMP does not provide. Biz Momentum is therefore
      3 metrics, not 4, and the model carries no forward-looking weight in that
      pillar: it sees what companies reported, never what analysts now expect.
      Metric count 21 -> 20.

  HISTORY = 5 years everywhere, not 10. Consequence worth knowing: the
      drawdown record no longer reaches the 2020 crash, so "worst winter on
      record" for most names is 2022. V5 compares against a 5-year median.

  PERIOD-END dates, not filing dates. Fine for live screening. Any backtest
      built on this is void (no point-in-time data).

  NO consensus snapshots, no stored history, no AI analyst layer.

  NOPAT uses each company's own effective tax rate clamped to 0-35%, with a
      21% fallback only where no rate can be computed (see NOPAT_TAX_RATE).

  US universe only (SPY is the benchmark).

  Financials variant uses FMP-derivable proxies for PPOP / NIM / GWP /
      combined ratio — see metrics.py. Provisioning quality remains invisible.
"""
from __future__ import annotations

# Percentiles are always relative to the loaded universe - the list of tickers
# passed to vqmb.run() - so a name can sit at 90 in one list and 60 in another.
# That is the point, not a defect.
HISTORY_YEARS = 5

# --------------------------------------------------------------------------
# Profiles (handbook §9) — frozen, no sliders, ever.
# --------------------------------------------------------------------------
PROFILES = {
    "TRADER": {"V": 20.0, "Q": 20.0, "B": 20.0, "P": 40.0},   # tape-led, weeks
    "PM":     {"V": 37.5, "Q": 25.0, "B": 25.0, "P": 12.5},   # value-led, years
    "GROWTH": {"V": 10.0, "Q": 25.0, "B": 40.0, "P": 25.0},   # acceleration-led, quarters
}
BASE_PROFILE = "TRADER"

# --------------------------------------------------------------------------
# Pillar structure. Each metric is (key, direction) where direction is
# +1 "higher raw ranks higher" or -1 "lower raw ranks higher" (the inv marker).
# Sub-blocks matter: a blank reweights only among its own siblings (§13).
# --------------------------------------------------------------------------
VALUE_METRICS = [
    ("v1_ebit_ev", +1),
    ("v2_ev_gp", -1),          # FIN variant flips to book/price, direction +1
    ("v3_fwd_earn_yield", +1),
    ("v4_norm_ep", +1),
    ("v5_ev_sales_vs_hist", -1),
]
QUALITY_ENGINE = [("q1_roic", +1), ("q2_roiic", +1), ("q3_persistence", +1)]
QUALITY_SHIELD = [("q4_leverage_score", +1), ("q5_drawdown", -1), ("q6_hygiene", -1)]
BIZ_METRICS = [("b1_vs_trend", +1), ("b2_sequential", +1), ("b3_margin_delta", +1)]
PRICE_STRENGTH = [("p1_trend_12_1", +1), ("p2_trend_6_1", +1), ("p3_high_distance", +1)]
PRICE_CREDIBILITY = [("p4_continuity", +1), ("p5_lottery", -1), ("p6_down_resilience", +1)]

# metrics whose direction flips under the financials variant
FIN_DIRECTION_OVERRIDE = {"v2_ev_gp": +1}      # becomes book/price

# Metrics whose FORMULA differs for financials (§12). These are ranked within
# the FIN group only and the resulting percentile is spliced into the U column,
# because the two variants measure different quantities: a bank's ROE is not
# comparable to an industrial's ROIC, and its capital ratio is not a leverage
# score. Ranking them on one scale silently compares apples to oranges.
#
# Everything NOT listed here is identical under both variants - V3, Q5 and all
# of P1-P6 - and so ranks universe-wide as normal.
VARIANT_METRICS = {
    "v1_ebit_ev",          # EBIT/EV        -> E/P
    "v2_ev_gp",            # EV/GP          -> book/price
    "v4_norm_ep",          # normalized E/P -> 5y avg ROE x book / mcap
    "v5_ev_sales_vs_hist", # EV/S vs own    -> P/B vs own
    "q1_roic",             # ROIC           -> ROE
    "q2_roiic",            # ROIIC          -> incremental ROE
    "q3_persistence",      # on revenue     -> on PPOP / GWP
    "q4_leverage_score",   # ND/EBIT curve  -> equity/assets curve
    "q6_hygiene",          # accruals dropped, bloat redefined
    "b1_vs_trend",         # volume line    -> PPOP / GWP
    "b2_sequential",
    "b3_margin_delta",     # gross margin   -> NIM / combined ratio
}

ALL_METRICS = (VALUE_METRICS + QUALITY_ENGINE + QUALITY_SHIELD + BIZ_METRICS
               + PRICE_STRENGTH + PRICE_CREDIBILITY)
METRIC_COUNT = len(ALL_METRICS)                # 20

# --------------------------------------------------------------------------
# Dampeners (§6, §8) — multiply inside a pillar, before re-ranking.
# Both can only ever COST score, never add: floor + span * (x/100) where
# floor + span = 1.0 at x = 100.
# --------------------------------------------------------------------------
SHIELD_FLOOR, SHIELD_SPAN = 0.6, 0.4        # quality:  0.6 + 0.4 * shield/100
CRED_FLOOR, CRED_SPAN = 0.5, 0.5            # price:    0.5 + 0.5 * cred/100

# --------------------------------------------------------------------------
# Gates and overlays (§11) — outside every average, never bought back.
# --------------------------------------------------------------------------
LEVERAGE_GATE_ND_EBIT = 4.0                 # net debt / EBIT above this -> GATED
BANK_CAPITAL_GATE = 0.05                    # equity / assets below this -> GATED
INSURER_CAPITAL_GATE = 0.08                 # equity / assets proxy for solvency

# Q4 leverage score curve (§6): net cash -> 100; 0..3x linear 100->60;
# 3..4x convex 60 -> 0; above 4x -> 0 and gate.
LEV_KINK, LEV_MAX = 3.0, 4.0
LEV_SCORE_AT_KINK = 60.0

# --------------------------------------------------------------------------
# Conviction (§10)
# --------------------------------------------------------------------------
GREEN_TOP_PCT = 5.0            # top 5% of the active profile, AND...
GREEN_DISPERSION_FLOOR = 8.0   # ...raw composite this far above universe median.
                               # PLACEHOLDER: the handbook says "tune in
                               # backtest". Backtests need point-in-time data,
                               # which this build does not have. Do not read 8
                               # as a considered number.
TRIPLE_TOP_PCT = 10.0          # top decile on all three profiles -> the star

# --------------------------------------------------------------------------
# Flags with hysteresis (§16). ON threshold, then a slacker OFF threshold, so a
# name sitting on a boundary does not blink on and off night after night.
# Screen shows at most 2, in this severity order; the card shows all.
# --------------------------------------------------------------------------
FLAG_SEVERITY = ["GATE", "FRAGILE", "FX", "THIN", "STALE", "HYGIENE", "DISCRETE",
                 "CONFLICT", "SHORT_HISTORY", "LOSS", "CALC", "NOGP", "GPA", "PXCHK",
                 "WKCHK", "FIN"]
SCREEN_FLAG_LIMIT = 2

HYGIENE_ON, HYGIENE_OFF = 20.0, 30.0
DISCRETE_STRENGTH_ON, DISCRETE_CRED_ON = 67.0, 33.0
DISCRETE_CRED_OFF, DISCRETE_STRENGTH_OFF = 45.0, 55.0
# Raised from the handbook's 60. For four independent percentile pillars the
# MEDIAN spread is 61.5, so a bar of 60 fired on 55% of the index and marked
# nothing. 75 fires on roughly a quarter - names where the lenses genuinely
# disagree. The OFF bar keeps the same 10-point hysteresis gap.
CONFLICT_SPREAD_ON, CONFLICT_SPREAD_OFF = 75.0, 65.0

# --------------------------------------------------------------------------
# Safety grade (§16) — display only, ABSOLUTE by design (a fortress is a
# fortress regardless of peers). Contrast with A/D, which is relative.
# --------------------------------------------------------------------------
SAFETY_SHIELD_W, SAFETY_DRAWDOWN_W = 0.6, 0.4
SAFETY_BANDS = [(85, "A"), (70, "B"), (50, "C"), (30, "D"), (0, "E")]

# --------------------------------------------------------------------------
# Groups and sector ranks (§14)
# --------------------------------------------------------------------------
# The handbook asks for two things that pull against each other on a 500-name
# universe: 20-24 US groups (§14) AND a 15-member floor. Enforcing the floor
# strictly folds Transport, Banks, Aerospace and Commercial Services away and
# leaves 16 groups, with Capital Goods at 71 names - fewer groups than asked
# for, and a bucket too coarse to read. A floor of 12 merges only Commercial
# Services and leaves 19 groups with a largest of 46.
#
# 12 is the setting. It is a deliberate deviation from the stated floor, taken
# because the group-count target is the more informative of the two: the layer
# exists to separate the tide from the swimmer, and a 71-name "Capital Goods"
# is not a tide anybody trades. Raise this to 15 to follow the floor instead.
MIN_GROUP_SIZE = 12            # smaller merges to nearest neighbour (§14)

# --------------------------------------------------------------------------
# Data minima (§17)
# --------------------------------------------------------------------------
MOM_MIN_OBS_12_1 = 9           # of 11 monthly observations
MOM_MIN_OBS_6_1 = 4            # of 5
DOWN_DAYS_MIN = 8              # qualifying benchmark down-days for P6
DRAWDOWN_MIN_YEARS = 4         # relaxed from the handbook's 5 because this
                               # build only pulls 5 years of history at all
PERSISTENCE_MIN_QUARTERS = 4   # of 8
ENTRY_MIN_PRICE_YEARS = 1      # universe entry rule (§4)

DRAWDOWN_EPISODE = -0.30       # episode opens at -30% from peak...
DRAWDOWN_RECOVERY = 0.90       # ...and closes on recovery to within 10% of it
MAX_LOTTERY_DAYS = 5           # P5 averages the 5 largest daily returns
# NOPAT tax. Fixed rate rather than each company's effective rate, as in the
# previous model: a flat multiplier keeps the cross-section comparable and stops
# one-off tax items (settlements, valuation-allowance releases, repatriation
# charges) from moving a company's apparent return on capital. The handbook
# specifies the effective rate clamped 0-35%; this is a deliberate deviation.
# US-only, so 21% is the statutory federal rate.
NOPAT_TAX_RATE = 0.21          # fallback ONLY, where a company's effective rate
                               # cannot be computed (no pre-tax income line)
TAX_CLAMP = (0.0, 0.35)        # handbook §6: effective tax, clamped to this band
ROIIC_MIN_DELTA_IC = 0.05      # ΔIC below 5% of IC -> blank by inapplicability

# --------------------------------------------------------------------------
# Checklist (§16) — pinned to TRADER for every user, by design.
# --------------------------------------------------------------------------
CHECKLIST = [
    # the checklist is pinned to the base profile, so RNK reads that profile's
    # rank and GRP reads group strength computed for it (row 3)
    ("RNK >= 75", f"composite_rank_{BASE_PROFILE}", 75.0),
    ("GRP >= 60", "grp_base", 60.0),
    ("Value S-rank >= 60", "v_rank_s", 60.0),
    ("Engine >= 70", "quality_engine", 70.0),
    # leverage_ok is 100 (pass) or 0 (fail), so a bar of 50 can actually fail (row 5)
    ("Net cash or ND/EBIT < 1", "leverage_ok", 50.0),
    ("B >= 70", "b_rank_u", 70.0),
    ("Revisions >= 60", None, 60.0),      # B4 removed -> permanently NEUTRAL
    ("P >= 70", "p_rank_u", 70.0),
    ("Credibility >= 50", "price_credibility", 50.0),
    ("Hygiene >= 40", "q6_hygiene_pct", 40.0),
]
CHECKLIST_NEUTRAL_BAND = 10.0   # within 10 points of the bar -> NEUTRAL, not FAIL

# --------------------------------------------------------------------------
# Darvas state machine (§15) — unscored classification
# --------------------------------------------------------------------------
DARVAS_CONFIRM_SESSIONS = 3
DARVAS_BREAKOUT_VOLUME = 1.5    # x 50-day average
DARVAS_ELIGIBLE_PCT = 0.05      # must be within 5% of the 52-week high
DARVAS_MAX_BOX_HEIGHT = 0.20
DARVAS_DRIFT_SESSIONS = 60

AD_SESSIONS = 50                # up/down volume ratio window


# --------------------------------------------------------------------------
# Market vitals - how each is explained on the screen, and how it is read.
#
# The thresholds are INTERPRETIVE, not part of the scoring: they decide the
# word shown beside each number, never a rank. `bands` run low to high; each is
# (upper bound, verdict, tone), with None meaning "and above". Tone is good,
# note or bad and only colours the verdict.
# --------------------------------------------------------------------------
VITALS = {
    "max5": {
        "label": "Typical big-day move",
        "unit": "%",
        "what": "How much of a typical stock's year came in its five biggest up-days.",
        "why": "High means returns arrive in bursts - crowded, news-driven, prone to reversal.",
        "bands": [(5.0, "Calm", "good"), (8.0, "Normal", "note"), (None, "Jumpy", "bad")],
    },
    "continuity": {
        "label": "Up-days minus down-days",
        "unit": "pts",
        "what": "For the typical stock, the share of up-days less the share of down-days.",
        "why": "Positive means trends are building in small, steady steps - the kind that persist.",
        "bands": [(0.0, "Mostly down-days", "bad"), (5.0, "Mild uptrend", "note"),
                  (None, "Steady uptrend", "good")],
    },
    "nd_ebit": {
        "label": "Typical debt load",
        "unit": "x",
        "what": "Median net debt as a multiple of operating profit, banks excluded.",
        "why": "Years of profit it would take to repay borrowings. Above 4x a name is gated.",
        "bands": [(1.0, "Light", "good"), (3.0, "Moderate", "note"), (None, "Heavy", "bad")],
    },
    "gated": {
        "label": "Excluded by a gate",
        "unit": "%",
        "what": "Share of the universe barred from conviction by leverage or capital limits.",
        "why": "Gated names keep their rank but can never be green. Mostly utilities and REITs.",
        "bands": [(15.0, "Few", "good"), (30.0, "Typical", "note"), (None, "Many", "bad")],
    },
    "dispersion": {
        "label": "How far apart the scores are",
        "unit": "pts",
        "what": "The spread of the middle half of composite scores.",
        "why": "Wide means the model sees clear winners and losers; narrow means little to choose between names.",
        "bands": [(15.0, "Flat - few clear leaders", "bad"), (25.0, "Normal", "note"),
                  (None, "Wide - strong differentiation", "good")],
    },
    "corr": {
        "label": "Are strong trends orderly?",
        "unit": "",
        "what": "Correlation between each stock's momentum and how steadily it was earned.",
        "why": "Negative means the strongest names got there in jumps - the whole tape is fragile.",
        "bands": [(-0.2, "Fragile - strength came in jumps", "bad"), (0.2, "Neutral", "note"),
                  (None, "Orderly - strength is steady", "good")],
    },
    "green": {
        "label": "High-conviction names",
        "unit": "",
        "what": "Names in the top 5% of the active profile that also sit clearly above the median.",
        "why": "Capped at the top 5% by design, so read it against the ceiling: few means the leaders are not separating from the pack.",
        "bands": [],          # a count, read against the universe size rather than a band
    },
}


# --------------------------------------------------------------------------
# Flags on the screen: a short code in a chip, the full meaning on hover.
# (code, tone, meaning). Tone colours the chip: bad = red, warn = amber,
# note = grey.
# --------------------------------------------------------------------------
FLAG_CHIPS = {
    "GATE": ("GATE", "bad", "Gated - excluded from high conviction"),
    "HYGIENE": ("HYG", "warn", "Accounting tripwire - accruals, dilution or balance-sheet bloat"),
    "DISCRETE": ("JMP", "warn", "Jumpy tape - strong momentum earned in jumps, not steps"),
    "CONFLICT": ("MIX", "warn", "Pillars disagree - read the card, not the rank"),
    "LOSS": ("LOSS", "warn", "Consensus forecasts a loss"),
    "SHORT_HISTORY": ("NEW", "note", "Short history - too little data to know how it breaks"),
    "CALC": ("EST", "note", "No consensus - forward yield is model-built"),
    "FIN": ("FIN", "note", "Financials variant - ranked against financials only"),
    "NOGP": ("NO GP", "note", "No usable gross profit - the vendor reports no cost of revenue, "
                              "so EV/gross profit and margin change are set aside"),
    "GPA": ("GP/A", "note", "ROIC undefined - invested capital is zero or negative, so gross "
                            "profit / total assets stands in for return on capital"),
    "FX": ("FX", "bad", "Currency mismatch - statements and price in different currencies with "
                        "no exchange rate, so all five value lenses are set aside"),
    "STALE": ("STALE", "warn", "Stale or gapped data - newest statement over 200 days old, last "
                               "price over 7 days old, or reports not consecutive"),
    "THIN": ("THIN", "bad", "Thin data - fewer than half the pillars could be scored, so there is "
                            "no composite rank"),
    "PXCHK": ("PX?", "warn", "Last close differs from the quoted price by more than 10%"),
    "WKCHK": ("CHK", "bad", "A displayed calculation did not reproduce the ranked value"),
    "FRAGILE": ("FRG", "warn", "Country fragility"),
}


# --------------------------------------------------------------------------
# Data plausibility. Vendor statements are standardised into one template, and
# some businesses do not fit it: railroads, utilities and many service firms
# report no cost-of-revenue line, and the vendor then fills gross profit with
# the whole of revenue. That is not a 100% margin - it is a missing number, and
# treated as a number it ranks at the top of anything that uses it.
# --------------------------------------------------------------------------
GP_MAX_MARGIN = 0.995   # a gross margin at or above this is a filled-in total, not a margin


# ==========================================================================
# DATA-QUALITY REGISTER (Sep 2026). Each setting names the register row it
# implements. Switches that restore old behaviour are kept deliberately, so a
# rule can be rolled back without a code change.
# ==========================================================================

# row 1 - a loss-maker's ROIC is a real negative return, not a switch to a
# different ratio. "negative": EBIT <= 0 with IC > 0 -> EBIT / IC. "fallback":
# the old behaviour, gross profit / assets for any EBIT <= 0.
ROIC_LOSS_MODE = "negative"

# row 11 - a MISSING operating profit is missing data, not evidence of a loss.
# False: blank, no gate. A REPORTED EBIT <= 0 with net debt still gates.
MISSING_EBIT_GATES = False

# row 10 - one neutral value for a blank shield or credibility, used both in the
# pillar dampener and in the Safety grade, instead of best case in one place
# and worst case in the other
DAMPENER_NEUTRAL_FILL = 50.0

# A metric that is blank (missing or not computable) inside a pillar counts as
# a neutral percentile of 50 in its sub-block mean, instead of dropping out and
# re-averaging the rest. Applies to every sub-block of all four pillars (V,
# Q engine, Q shield, B, P strength, P credibility). The metric's own u_/s_
# percentile stays blank, so what was filled is still visible.
# Deliberate departure from FLUX, which drops the blank. False restores FLUX.
METRIC_NEUTRAL_FILL_ON = True
METRIC_NEUTRAL_FILL = 50.0

# row 6 - one line per issuer: dual share classes (GOOG/GOOGL) carry the same
# statements and would otherwise count twice in every percentile
FOLD_SHARE_CLASSES = True
DOLLAR_VOLUME_DAYS = 50

# row 30 - a composite needs at least this share of its pillars
COMPOSITE_MIN_PILLAR_SHARE = 0.5

# row 32 - abort the run if more than this share of names fail the workings check
WORKINGS_MISMATCH_ABORT_SHARE = 0.02

# row 8 - reporting cadence, detected from the gaps between report dates
CADENCE_QUARTERLY_DAYS = (70, 110)      # ~91 days between reports
CADENCE_HALFYEAR_DAYS = (150, 215)      # ~182 days between reports
YOY_PAIR_DAYS = (330, 400)              # every year-on-year pair must be this far apart

# row 28 - income and cash-flow statements must describe the same quarter
INCOME_CF_ALIGN_DAYS = 10

# row 33 - statement staleness
STALE_STATEMENT_DAYS = 200
STALE_BLANK_TTM = False                 # True also blanks the TTM-based metrics

# row 34 - price staleness and quote cross-check
STALE_PRICE_DAYS = 7
QUOTE_CLOSE_MAX_GAP = 0.10

# row 23 - the NTM fallback may only use a fiscal year that has not ended
NTM_FALLBACK_MIN_DAYS_AHEAD = 0

# rows 19-20 - every five-year average needs at least this many years
MIN_AVG_YEARS = 3

# row 37 - daily moves beyond this are reported for review, never dropped
IMPLAUSIBLE_DAILY_MOVE = 0.40          # 40%, not the register's 50%: a real 2-for-1 split day
                                        # only reaches -50% about half the time, as the day's own
                                        # move rides on the halving

# row 15 - dilution needs BOTH the share CAGR and stock comp / revenue.
# VERIFIED safe: companies with no stock comp arrive as 0, never null.
DILUTION_NEEDS_BOTH = True

# row 35 - VERIFIED against the cache (probe_vendor, Sep 2026): FMP sends no
# nulls on these fields and essentially never a 0 for revenue or total assets,
# so a 0 there is a missing value. Operating cash flow is treated as missing only
# when revenue is non-zero. Zero debt and zero goodwill are genuine and untouched.
ZERO_AS_MISSING = True
ZERO_AS_MISSING_FIELDS = ("revenue", "totalAssets")

# row 36 - VERIFIED against the cache: the estimates endpoint carries
# numAnalystsEps on every row. Fewer than MIN_ANALYSTS -> no consensus -> CALC.
MIN_ANALYSTS = 3
ANALYST_COUNT_FIELD = "numAnalystsEps"
