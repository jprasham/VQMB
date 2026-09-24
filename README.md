# VQMB

A quantitative stock-ranking model packaged as a Python library. It has four
sub-models:

| Pillar | Letter | What it measures |
|---|---|---|
| Value | **V** | Five valuation lenses; a name must look cheap several ways at once |
| Quality | **Q** | Returns on capital and growth persistence (the engine), discounted by a balance-sheet and drawdown shield |
| Business momentum | **B** | Acceleration in the business: revenue vs trend, sequential change, margin direction |
| Price momentum | **M** | Volatility-adjusted trend strength, discounted by a credibility dampener |

Twenty metrics are turned into percentiles and averaged into the four pillars.
The pillars are re-ranked, then combined three ways (the TRADER, PM and GROWTH
profiles) into composite ranks. Gates, conviction marks, grades and flags are
applied afterwards and never add score. Every rank runs from 0 to 100, and
**100 is best**.

You pass in a list of tickers and your FMP API key. You get back **one pandas
DataFrame** with one row per stock. It holds every source figure, metric,
percentile, pillar score, rank, gate, flag, grade and checklist item the
model produces.

---

## Install

```bash
pip install git+https://github.com/jprasham/VQMB.git
```

This needs Python 3.9 or later. The dependencies are `numpy`, `pandas` and
`requests`.

## Quick start

```python
import vqmb

df = vqmb.run(["AAPL", "MSFT", "NVDA", "JPM", "PGR", "XOM"], api_key="YOUR_FMP_KEY")

df[["symbol", "composite_rank_TRADER", "v_rank_u", "q_rank_u", "b_rank_u", "p_rank_u",
    "safety", "flags"]]
```

If you leave out `api_key`, the library reads the `FMP_API_KEY` environment
variable. Each developer uses their own key. The model reads FMP `/stable/`
endpoints, which are covered by the company's FMP Ultimate plan.

### Ranks are relative to your list

Every percentile and rank is computed **against the tickers you pass in**. A
stock can rank 90 in one list and 60 in another. That is how the model is
designed, not a bug. For the same ranks the S&P 500 screen shows, score the
whole index:

```python
tickers = vqmb.index_constituents("sp500")          # or "nasdaq", "dowjones", "etf:QQQ"
df = vqmb.run(tickers)                              # about 9 FMP calls per name
```

Small lists give coarse percentiles. A metric needs at least 2 names with a
value to be ranked. Sector ranks and the financials splice each need at least
12 names in the group; below that they fall back to one universe-wide ranking.

---

## API

### `vqmb.run(tickers, api_key=None, *, profile="TRADER", previous_flags=None, workers=8, cache_dir=None, refresh=False, include_price_history=False, include_source_data=False)`

Fetches everything from FMP and scores the list. Returns a DataFrame sorted by
the chosen profile's composite rank, best first.

| Argument | Meaning |
|---|---|
| `tickers` | A list of symbols, or one comma-separated string. They are upper-cased and de-duplicated. |
| `api_key` | Your FMP key. Falls back to `FMP_API_KEY`. |
| `profile` | `"TRADER"`, `"PM"` or `"GROWTH"`. Sets the sort order and which profile the `grp` / `group_*` columns use. All three composites are always computed. |
| `previous_flags` | The DataFrame from an earlier run, or any table with a `symbol` column plus flag columns. Gives the HYGIENE, DISCRETE and CONFLICT flags their hysteresis (see Flags). Without it, every flag uses its ON threshold. |
| `workers` | Number of parallel threads. |
| `cache_dir` | An optional folder for raw FMP responses. Fundamentals are reused for 1 day and market-cap history for 7 days. Prices are always fetched fresh. There is no caching by default. |
| `refresh` | Ignore cached fundamentals. |
| `include_price_history` | Adds a `price_history` column with the full daily series (dates, closes, highs, lows, volumes). |
| `include_source_data` | Adds a `source_data` column with the raw FMP responses (after aliasing and currency normalisation). |

It raises `vqmb.VQMBError` instead of returning numbers that can't be trusted:

- the SPY benchmark series comes back empty
- no name can be scored
- more than 2% of names fail the workings self-check

### Other functions

| Function | Use |
|---|---|
| `vqmb.fetch_data(tickers, api_key)` | Fetch only. Returns `{ticker: {"blob", "ohlcv", "mcap_hist"}}`. |
| `vqmb.fetch_benchmark(api_key)` | SPY closes as `{date: close}`. |
| `vqmb.run_from_data(data, benchmark, ...)` | Score data you already have, with no API calls. Takes the same keyword arguments as `run`. Use it to score one fetch more than once, or to score saved data. |
| `vqmb.index_constituents(index, api_key)` | Index or ETF members from FMP. |

Progress and warnings go to the standard `logging` logger named `"vqmb"`.

---

## The output DataFrame

One row per scored name. It has about 310 columns, grouped below. Blank
(`NaN`) always means "not computable". The model never fills a blank with
zero. A blank metric drops out, and the other metrics in its sub-block are
averaged without it.

### Headline

| Column | Meaning |
|---|---|
| `symbol`, `company`, `sector`, `industry` | Identity (FMP profile). |
| `group` | One of the industry groups, after merging any group with under 12 members into its neighbour. `group_raw` is the group before merging. |
| `composite_rank_TRADER` / `_PM` / `_GROWTH` | Composite rank, 0–100, for each profile. |
| `v_rank_u`, `q_rank_u`, `b_rank_u`, `p_rank_u` | Pillar ranks against the whole universe (U). |
| `v_rank_s`, `q_rank_s`, `b_rank_s`, `p_rank_s` | Pillar ranks within the sector (S). |
| `grp` | Group strength for the chosen profile. |
| `ad` | Accumulation/distribution grade A–E, by universe quintile. |
| `safety` | Safety grade A–E. Any gate forces E. |
| `darvas`, `darvas_stack` | Darvas box state: `BOX`, `BRK UP`, `BRK DOWN` or `DRIFT`, plus `STACK xN`. |
| `flags` | The two most severe active flags. |
| `gated`, `gate_cause` | Whether a gate applies, and why. |
| `checklist_passes` | Count of PASS results on the 10-item checklist. |
| `triple`, `green_TRADER` / `_PM` / `_GROWTH` | Conviction marks. |

### Metric raw values (the 20 ranked metrics)

| Pillar | Columns |
|---|---|
| Value | `v1_ebit_ev`, `v2_ev_gp`, `v3_fwd_earn_yield`, `v4_norm_ep`, `v5_ev_sales_vs_hist` |
| Quality – engine | `q1_roic`, `q2_roiic`, `q3_persistence` |
| Quality – shield | `q4_leverage_score`, `q5_drawdown`, `q6_hygiene` (with its components `hyg_accruals`, `hyg_dilution`, `hyg_bloat`) |
| Business momentum | `b1_vs_trend`, `b2_sequential`, `b3_margin_delta` |
| Price – strength | `p1_trend_12_1`, `p2_trend_6_1`, `p3_high_distance` |
| Price – credibility | `p4_continuity`, `p5_lottery`, `p6_down_resilience` |

Tags on each metric: `v3_calc`, `v3_loss`, `v3_calc_na`, `fwd_basis`,
`ntm_coverage`, `ntm_blend`, `q1_used_gpa_fallback`, `leverage_gate`,
`is_fin`, `fin_kind`.

### Percentiles

- `pct_<metric>` is the metric's universe percentile.
- `pct_<metric>__S` is its sector percentile.
- `pct_hyg_*_pct` are the three hygiene component percentiles.
- `q6_hygiene_pct` is the percentile of the hygiene composite.

### Pillars and composites

| Column | Meaning |
|---|---|
| `value_raw` | Mean of the V1–V5 percentiles. |
| `quality_engine`, `quality_shield` | Mean of the Q1–Q3 and Q4–Q6 percentiles. |
| `quality_damp`, `quality_raw` | `quality_damp = 0.6 + 0.4 × shield/100`, and `quality_raw = engine × quality_damp`. |
| `biz_raw` | Mean of the B1–B3 percentiles. |
| `price_strength`, `price_credibility` | Mean of the P1–P3 and P4–P6 percentiles. |
| `price_damp`, `price_raw` | `price_damp = 0.5 + 0.5 × credibility/100`, and `price_raw = strength × price_damp`. |
| `composite_raw_<PROFILE>` | Profile-weighted mean of the pillar U ranks. |
| `pillars_present`, `thin` | How many pillars were scored. `thin` means fewer than 2. |

### Overlays

| Column | Meaning |
|---|---|
| `safety_raw` | `0.6 × shield + 0.4 × drawdown percentile`. |
| `nd_ebit`, `ebit_negative_with_net_debt`, `capital_ratio` | Gate inputs. |
| `leverage_ok`, `grp_base` | Checklist inputs. |
| `chk::<item>` | PASS, NEUTRAL or FAIL for each of the 10 checklist items. |
| `flag_<NAME>` | Every flag as a boolean. `flag_pillar_spread` is the gap between the highest and lowest pillar rank. |
| `ad_ratio` | Raw up-volume ÷ down-volume. |
| `box_top`, `box_floor`, `days_in_box`, `tightness`, `stack` | Darvas box detail. |
| `group_n`, `group_median_composite`, `group_median_price_rank`, `group_iqr`, `group_strength` | The name's row of the group table. |

### Source figures and intermediate calculations (the fact sheet)

TTM and annual statement figures:

- `rev_ttm`, `gp_ttm`, `ebit_ttm`, `ni_ttm`, `cfo_ttm`, `sbc_ttm`
- `rev_ttm_prior`, `rev_a0`, `rev_3y`, `shares_a0`, `shares_3y`, `shares_now`

Balance sheet and valuation:

- `price`, `mcap`, `ev`, `net_debt`, `book`, `book_tangible`, `assets`,
  `assets_3y`, `invested_capital`

Tax, NOPAT and history:

- `effective_tax_rate`, `d_nopat_3y`, `d_ic_3y`, `d_ni_3y`, `d_book_3y`
- `net_margin_5y_avg`, `margin_years`, `roe_5y_avg`, `roe_years`

Forward estimates:

- `ntm_eps`, `fwd_rev`, `analyst_count`, `thin_consensus`, `g_projection`

Business-momentum inputs:

- `vol_yoy_q0`, `vol_yoy_q1`, `vol_cagr_3y`, `volume_quarters`
- `persistence_hits`, `persistence_quarters`
- `gm_q0`, `gm_q4`, `fin_margin_q0`, `fin_margin_q4`, `fin_margin_basis`

Own-history medians and price inputs:

- `ev_sales_median_hist` / `pb_median_hist`, `high_252`
- `monthly_returns_12_1`, `monthly_returns_6_1`, `daily_returns_252`
- `pct_up_days`, `pct_down_days`, `period_return_sign`, `returns_on_down_days`
- `dd_episodes`, `dd_worst`, `price_years`, `dollar_volume`

Data-quality fields:

- `cadence`, `ttm_n`, `yoy_lag`, `gapped`, `stale_statement`,
  `statement_age_days`, `price_age_days`
- `gp_quality`, `gp_set_aside`, `cf_aligned`, `currency`, `fx_mismatch`,
  `price_stale`, `stale`, `stale_reason`, `px_mismatch`, `implausible_moves`,
  `short_history`

Context columns (never ranked):

- `ctx_rev_cagr_3y`, `ctx_rev_yoy_ttm`, `ctx_rev_accel`, `ctx_gp_growth_ttm`
- `ctx_fwd_rev_growth`, `ctx_fwd_eps_growth`, `ctx_persistence`,
  `ctx_gross_margin`, `ctx_fcf_margin`

Statement lines behind each calculation, prefixed `w_`:

- The quarters in each TTM sum (`w_q`) and each year's margin and ROE
  (`w_margin_years`, `w_roe_years`)
- The fiscal years blended into NTM EPS (`w_ntm_parts`) and the historical
  EV/sales or P/B points (`w_hist_points`)

### Workings

`workings` holds a dict per name: one entry per metric, containing the formula,
every raw input, the result, and an `ok` field. `ok` records whether
recomputing the metric from its displayed inputs reproduces the ranked value.
`workings_mismatch` and `workings_mismatch_keys` summarise those checks, and
a mismatch raises the WKCHK flag.

### `df.attrs` (run-level information)

| Key | Meaning |
|---|---|
| `as_of`, `profile`, `names_in`, `names_scored` | Run identity. |
| `gated`, `green`, `triple`, `fin_variant` | Counts. |
| `failed` | `(ticker, error)` for names that could not be scored. |
| `excluded_entry_rule` | Names dropped for having under 1 year of prices. |
| `folded_share_classes` | Duplicate share classes that were folded (e.g. GOOG into GOOGL). |
| `merged_groups` | Groups under 12 members and the group each was merged into. |
| `blank_share` | Share of blanks for each metric. |
| `data_flags` | Counts of NOGP, GPA, FX, STALE, THIN, PXCHK, WKCHK and CALC. |
| `cadence` | Count of quarterly vs half-yearly reporters. |
| `workings_mismatch`, `implausible_moves` | Names listed for review. |
| `groups` | The group table: N, median composite, median price rank, IQR, GRP. |
| `market_vitals` | Seven universe-wide readings, each with a value and a verdict. |

---

## The model in brief

**Percentile.** `(rank − 1) / (n − 1) × 100`. Ties take the average rank. For
V2, V5, Q5 and P5, lower raw values are better. Banks, lenders and insurers
use a variant formula for 12 of the metrics, and those metrics rank
financials only against other financials. The two rankings are then spliced
into one column.

**Pillars.** A blank metric is removed, and the remaining metrics in its
sub-block are averaged without it. Weight never moves across sub-blocks.
Quality = engine × (0.6 + 0.4 × shield/100). Price = strength × (0.5 + 0.5 ×
credibility/100). Both dampeners can only reduce a score, and a blank shield
or blank credibility counts as a neutral 50.

**Profiles.** These are the pillar weights, in %:

| Profile | V | Q | B | M |
|---|---|---|---|---|
| TRADER | 20 | 20 | 20 | 40 |
| PM | 37.5 | 25 | 25 | 12.5 |
| GROWTH | 10 | 25 | 40 | 25 |

A name needs at least 2 of its 4 pillars to get a composite.

**Gates.** A name is gated if any of these apply:

- ND/EBIT above 4×
- a reported operating loss while carrying net debt
- for banks and lenders, equity/assets below 5%; for insurers, below 8%

A gated name keeps its ranks, but it can never be green or triple, and its
Safety grade is forced to E.

**Conviction.** A name is **green** when all three hold: it is in the top 5%
of the profile, its composite is at least 8 points above the universe
median, and it is not gated. It gets a **triple** when it is in the top
decile on all three profiles and is not gated.

**Flags.** Flags never change a score.

- GATE, FX, THIN, STALE, HYGIENE, DISCRETE, CONFLICT, LOSS, PXCHK,
  SHORT_HISTORY, CALC, NOGP, GPA, FIN, WKCHK.
- HYGIENE, DISCRETE and CONFLICT have hysteresis: once on, a flag stays on
  until a looser OFF threshold is crossed. Pass the previous run's output as
  `previous_flags` to carry that state forward.

All constants live in [`vqmb/config.py`](vqmb/config.py).

---

## Development

```bash
pip install -e ".[test]"
pytest
```

The tests include:

- the reference worked example: engine 75 / shield 78 → quality 68.4, and
  composites of 70 / 74.625 / 75.7
- one check per data-quality rule
- a full synthetic-universe run through the engine

None of them need an API key.
