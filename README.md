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
DataFrame** with one row per stock and 146 columns: identity, the 20 metrics,
their universe and sector percentiles, blocks and pillars, composites,
industry group, overlays, the checklist and context.

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

df[["symbol", "rank", "V", "Q", "B", "P", "gated", "flags", "safety_grade"]]
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

### `vqmb.run(tickers, api_key=None, *, profile="TRADER", previous_flags=None, workers=8, cache_dir=None, refresh=False)`

Fetches everything from FMP and scores the list. Returns the 146-column
DataFrame, sorted by `rank`, best first.

| Argument | Meaning |
|---|---|
| `tickers` | A list of symbols, or one comma-separated string. They are upper-cased and de-duplicated. |
| `api_key` | Your FMP key. Falls back to `FMP_API_KEY`. |
| `profile` | `"TRADER"`, `"PM"` or `"GROWTH"`. The active profile: it fills `composite`, `rank` and `green`, and sets the sort order. All three composites are always computed. |
| `previous_flags` | The DataFrame from an earlier run (its `flags` column is read). Gives the HYGIENE, DISCRETE and CONFLICT flags their hysteresis (see Flags). Without it, every flag uses its ON threshold. |
| `workers` | Number of parallel threads. |
| `cache_dir` | An optional folder for raw FMP responses. Fundamentals are reused for 1 day and market-cap history for 7 days. Prices are always fetched fresh. There is no caching by default. |
| `refresh` | Ignore cached fundamentals. |

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

`vqmb.run()` returns exactly **146 columns**, one row per scored name, sorted
by `rank` (best first). The list is `vqmb.engine.COLUMNS`. Blank (`NaN`)
always means "not computable". The model never fills a blank with zero. A
blank metric keeps a blank `u_`/`s_` percentile, but counts as a neutral 50
inside its sub-block's average (see `neutral_fill_metrics`). Flag-like columns are `1`/`0`. Multi-value text columns are
pipe-separated.

### Identity (8)

`symbol`, `company`, `sector`, `industry`, `country`, `price` (latest close),
`mktcap`, `fin_mode`.

`fin_mode` is `1` when the financials variant applies. That covers banks,
lenders and insurers only. Exchanges, asset managers, brokers and payment
networks stay on the standard set.

### Ranked metrics, raw values (20)

| Block | Columns |
|---|---|
| Value | `ebit_to_ev`, `ev_to_gp`↓, `fwd_earn_yield`, `normalized_ep`, `ev_sales_vs_hist`↓ |
| Quality – engine | `roic`, `roiic`, `growth_persistence` |
| Quality – shield | `leverage_score`, `drawdown_history`↓, `hygiene` |
| Business momentum | `latest_q_vs_3y`, `seq_accel`, `gm_change_yoy` |
| Price – strength | `trend_12_1`, `trend_6_1`, `dist_from_high` |
| Price – credibility | `continuity`, `lottery_days`↓, `down_resilience` |

↓ = lower is better. For `ev_to_gp`, the financials variant is book/price,
and there higher is better.

### Percentiles (40)

- `u_<metric>`: the universe percentile, 0–100, where 100 is always best.
  This is what feeds the score. For the 12 variant metrics, financials are
  ranked only against other financials.
- `s_<metric>`: the sector percentile. It is diagnostic only. Sectors with
  fewer than 12 names fall back to the universe percentile. For financials,
  `s_` equals `u_` by construction on the variant metrics.

### Blocks and pillars (20)

| Column | Meaning |
|---|---|
| `blk_value` | Mean of the V1–V5 percentiles. In every `blk_` column a blank metric counts as 50. |
| `blk_q_engine` | Mean of the Q1–Q3 percentiles. |
| `blk_q_shield` | Mean of the Q4–Q6 percentiles. |
| `blk_b_mom` | Mean of the B1–B3 percentiles. |
| `blk_p_strength` | Mean of the P1–P3 percentiles. |
| `blk_p_cred` | Mean of the P4–P6 percentiles. |
| `shield_dampener` | `0.6 + 0.4 × shield/100`. A blank shield counts as 50. |
| `credibility_dampener` | `0.5 + 0.5 × credibility/100`. A blank credibility counts as 50. |
| `V_raw`, `B_raw` | Equal to `blk_value` and `blk_b_mom`. |
| `Q_raw` | `blk_q_engine × shield_dampener`. |
| `P_raw` | `blk_p_strength × credibility_dampener`. |
| `V`, `Q`, `B`, `P` | The pillar raw scores re-ranked 0–100 against the universe. |
| `s_V`, `s_Q`, `s_B`, `s_P` | The same pillar raw scores re-ranked 0–100 within the name's sector. Diagnostic only; never feeds the composite. |

### Composite (9)

| Column | Meaning |
|---|---|
| `profile` | The active profile, set by `run(profile=...)`. |
| `composite`, `rank` | The active profile's weighted score, and that score re-ranked 0–100. |
| `composite_TRADER` / `_PM` / `_GROWTH` | Weighted score for each profile. |
| `rank_TRADER` / `_PM` / `_GROWTH` | Each profile's score re-ranked 0–100. |

A name needs at least 2 of the 4 pillars to get a composite. Otherwise it is
blank and flagged THIN. With the neutral fill on, a pillar is never blank (a
sub-block with every metric blank scores 50), so THIN does not fire.

### Industry group (4)

| Column | Meaning |
|---|---|
| `group_raw` | The industry group the taxonomy assigns from sector and industry. |
| `group` | The group after merges: a group under 12 members is folded into its nearest neighbour. |
| `grp` | Group strength for the active profile: the median member `rank`, re-ranked 0–100 across groups, rounded. Matches `df.attrs["groups"]`. |
| `grp_base` | Group strength for the TRADER profile, unrounded. This is what the checklist reads. |

### Overlays (9)

| Column | Meaning |
|---|---|
| `gated` | `1` when a gate applies. |
| `gate_cause` | Why the name is gated, exactly as `ranking.apply_gates` records it (the first rule that fired): `ND/EBIT > 4x`, `EBIT <= 0 with net debt`, `capital below floor` (banks and lenders < 5%, insurers < 8% equity/assets) or `leverage gate (Q4)`. Blank when not gated. |
| `flags` | Every active flag, most severe first: FX, THIN, STALE, HYGIENE, DISCRETE, CONFLICT, SHORT_HISTORY, LOSS, CALC, NOGP, GPA, PXCHK, WKCHK, FIN. |
| `pillar_spread` | Highest pillar rank minus lowest. |
| `safety_score` | `0.6 × shield + 0.4 × drawdown percentile`. |
| `safety_grade` | A–E on absolute bands (85 / 70 / 50 / 30). Any gate forces E. |
| `ad_grade` | A–E by universe quintile of `ad_ratio`. |
| `green` | `1` = top 5% of the active profile, at least 8 points above the median composite, and not gated. |
| `triple` | `1` = top decile on all three profiles, and not gated. |

### Checklist (12)

Ten items, each `PASS`, `NEUTRAL` (within 10 points below the bar, or no data)
or `FAIL`. The checklist is pinned to the TRADER profile whatever `profile` is
set. Column names come from `config.CHECKLIST`.

| Column | Reads | Bar |
|---|---|---|
| `chk::RNK >= 75` | TRADER composite rank | 75 |
| `chk::GRP >= 60` | `grp_base` | 60 |
| `chk::Value S-rank >= 60` | `s_V` | 60 |
| `chk::Engine >= 70` | `blk_q_engine` | 70 |
| `chk::Net cash or ND/EBIT < 1` | `leverage_ok` | 50 |
| `chk::B >= 70` | `B` | 70 |
| `chk::Revisions >= 60` | none - always `NEUTRAL` | 60 |
| `chk::P >= 70` | `P` | 70 |
| `chk::Credibility >= 50` | `blk_p_cred` | 50 |
| `chk::Hygiene >= 40` | `u_hygiene` | 40 |
| `checklist_passes` | Number of items at `PASS`, 0–10. | |
| `leverage_ok` | `100` = net cash or ND/EBIT under 1; `0` = ND/EBIT of 1 or more, or a loss with net debt; blank for financials or missing data. | |

### Context, never ranked (24)

| Column | Meaning |
|---|---|
| `ev` | Market cap + total debt − cash & short-term investments. |
| `net_debt` | Total debt − cash & short-term investments. |
| `equity_to_assets` | The capital ratio. |
| `ntm_coverage` | How much of the next 12 months the estimate rows span. |
| `fwd_basis` | `ntm` = blended across fiscal years. `fy_fallback` = coverage under 80%, so the nearest fiscal year that hasn't ended was used. `none` = no usable consensus (including fewer than 3 analysts). |
| `fwd_rev_basis` | The same basis flag, for revenue. |
| `ntm_blend` | The fiscal-year weights actually used, e.g. `31% FY2026 + 69% FY2027`. |
| `fwd_rev_ntm` | Blended next-twelve-month revenue. |
| `fwd_eps_ntm` | Blended next-twelve-month EPS. |
| `fwd_rev_growth` | NTM revenue against TTM revenue. |
| `fwd_eps_growth` | NTM EPS against TTM EPS. |
| `fwd_pe` | Price ÷ NTM EPS. Blank when the EPS is ≤ 0. |
| `ad_ratio` | Up-day volume ÷ down-day volume over 50 sessions. |
| `rev_yoy_ttm` | TTM revenue growth; `g` in the modelled forward yield. |
| `fwd_earn_yield_calc` | `0` = consensus. `1` = modelled, because there was no usable consensus (the value can still be blank). `2` = blank because consensus forecasts a loss. |
| `net_debt_to_ebit` | Net debt ÷ TTM EBIT. Negative means net cash. Blank when EBIT ≤ 0. |
| `accruals`, `dilution`, `bs_bloat` | The three raw hygiene components. For all three, higher is worse. |
| `rev_cagr_3y`, `rev_yoy_q0` | The two legs of `latest_q_vs_3y`, measured on the volume line: revenue for most companies, pre-provision profit for banks and lenders. |
| `short_history` | `1` when a history minimum isn't met. |
| `short_history_metrics` | Which of `drawdown_history`, `growth_persistence` and `ev_sales_vs_hist` are short of history. |
| `neutral_fill_metrics` | The ranked metrics that were blank and counted as a neutral 50 in their sub-block. Empty when all 20 are present. |

### `df.attrs` (run-level information, not columns)

| Key | Meaning |
|---|---|
| `as_of`, `profile`, `names_in`, `names_scored` | Run identity. |
| `gated`, `green`, `triple`, `fin_variant` | Counts. |
| `failed` | `(ticker, error)` for names that could not be scored. |
| `excluded_entry_rule` | Names dropped for having under 1 year of prices. |
| `folded_share_classes` | Duplicate share classes that were folded (e.g. GOOG into GOOGL). |
| `merged_groups` | Industry groups under 12 members and the group each was merged into. |
| `blank_share` | Share of blanks for each metric. |
| `data_flags` | Counts of NOGP, GPA, FX, STALE, THIN, PXCHK, WKCHK and CALC. |
| `cadence` | Count of quarterly vs half-yearly reporters. |
| `workings_mismatch`, `implausible_moves` | Names listed for review. |
| `groups` | The industry-group table. |
| `market_vitals` | Seven universe-wide readings, each with a value and a verdict. |

---

## The model in brief

**Percentile.** `(rank − 1) / (n − 1) × 100`. Ties take the average rank. For
V2, V5, Q5 and P5, lower raw values are better. Banks, lenders and insurers
use a variant formula for 12 of the metrics, and those metrics rank
financials only against other financials. The two rankings are then spliced
into one column.

**Pillars.** A blank metric counts as a neutral percentile of 50, and every
sub-block is the equal-weight mean of all its metrics. Example: B1 blank, B2 at
80, B3 at 60 → business momentum = (50 + 80 + 60) / 3 = 63.3. Weight never
moves across sub-blocks. This differs from FLUX, which drops a blank and
averages the rest; `config.METRIC_NEUTRAL_FILL_ON = False` restores the FLUX
rule.
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
