"""
VQMB group taxonomy (handbook §14).

Maps FMP industry strings to roughly GICS industry-group granularity — the
handbook wants 20-24 groups for the US with a minimum of 15 members each, and
the go/no-go gate wants at least 98% of names mapped.

The group layer is unscored. It exists so a PM can separate the tide from the
swimmer: RNK 88 with GRP 91 is a group bet; RNK 88 with GRP 30 is idiosyncratic
alpha against a headwind.

Unmapped strings fall into UNMAPPED, which is surfaced and counted rather than
hidden — an unmapped name gets a blank sector rank, a blank GRP, and a NEUTRAL
on checklist item 2.

DRAFT. Built from the S&P 500's actual FMP industry strings. Worth a review
before it is treated as settled, particularly where a group sits near the
15-member floor.
"""
from __future__ import annotations

GROUPS: dict[str, tuple[str, ...]] = {
    "Semiconductors": ("semiconductor", "semiconductors", "semiconductor equipment & materials"),
    "Software": ("software - infrastructure", "software - application", "software"),
    "Tech Hardware": ("computer hardware", "consumer electronics", "electronic components",
                      "communication equipment", "information technology services",
                      "scientific & technical instruments", "electronics & computer distribution",
                      "solar"),
    "Media & Entertainment": ("entertainment", "broadcasting", "advertising agencies",
                              "publishing", "electronic gaming & multimedia", "internet content & information"),
    "Telecom": ("telecom services",),
    "Retail": ("specialty retail", "internet retail", "apparel retail", "home improvement retail",
               "auto & truck dealerships", "department stores", "luxury goods",
               "discount stores", "footwear & accessories", "apparel manufacturing"),
    "Consumer Staples": ("grocery stores", "food distribution", "packaged foods",
                         "beverages - non-alcoholic", "beverages - brewers",
                         "beverages - wineries & distilleries", "household & personal products",
                         "tobacco", "confectioners", "farm products"),
    "Restaurants & Leisure": ("restaurants", "lodging", "resorts & casinos", "travel services",
                              "leisure", "gambling", "personal services"),
    "Healthcare Equipment": ("medical devices", "medical instruments & supplies",
                             "diagnostics & research", "health information services"),
    "Pharma & Biotech": ("drug manufacturers - general", "drug manufacturers - specialty & generic",
                         "biotechnology"),
    "Healthcare Services": ("healthcare plans", "medical care facilities",
                            "pharmaceutical retailers", "medical distribution"),
    "Banks": ("banks - diversified", "banks - regional", "banks", "savings & cooperative banks",
              "mortgage finance"),
    "Diversified Financials": ("credit services", "capital markets", "financial data & stock exchanges",
                               "asset management", "financial conglomerates", "shell companies"),
    "Insurance": ("insurance - diversified", "insurance - property & casualty", "insurance - life",
                  "insurance - specialty", "insurance - reinsurance", "insurance brokers"),
    "Real Estate": ("reit - industrial", "reit - office", "reit - retail", "reit - residential",
                    "reit - healthcare facilities", "reit - hotel & motel", "reit - specialty",
                    "reit - diversified", "reit - mortgage", "real estate services",
                    "real estate - development", "real estate - diversified"),
    "Capital Goods": ("specialty industrial machinery", "farm & heavy construction machinery",
                      "industrial distribution", "building products & equipment",
                      "engineering & construction", "electrical equipment & parts",
                      "infrastructure operations", "tools & accessories", "metal fabrication",
                      "pollution & treatment controls", "business equipment & supplies"),
    "Aerospace & Defense": ("aerospace & defense",),
    "Transport": ("railroads", "trucking", "integrated freight & logistics", "airlines",
                  "marine shipping", "airports & air services"),
    "Commercial Services": ("specialty business services", "staffing & employment services",
                            "consulting services", "security & protection services",
                            "rental & leasing services", "waste management",
                            "education & training services"),
    "Autos": ("auto manufacturers", "auto parts", "recreational vehicles"),
    "Energy": ("oil & gas integrated", "oil & gas e&p", "oil & gas midstream",
               "oil & gas refining & marketing", "oil & gas equipment & services",
               "oil & gas drilling", "thermal coal", "uranium"),
    "Materials": ("specialty chemicals", "chemicals", "agricultural inputs", "copper", "gold",
                  "steel", "aluminum", "other industrial metals & mining", "paper & paper products",
                  "packaging & containers", "building materials", "lumber & wood production",
                  "silver", "other precious metals & mining", "coking coal"),
    "Utilities": ("utilities - regulated electric", "utilities - regulated gas",
                  "utilities - regulated water", "utilities - diversified",
                  "utilities - renewable", "utilities - independent power producers"),
}

# reverse index, lowercased
_LOOKUP = {ind: g for g, inds in GROUPS.items() for ind in inds}

SECTOR_FALLBACK = {
    "technology": "Software",
    "communication services": "Media & Entertainment",
    "consumer cyclical": "Retail",
    "consumer defensive": "Consumer Staples",
    "healthcare": "Healthcare Services",
    "financial services": "Diversified Financials",
    "real estate": "Real Estate",
    "industrials": "Capital Goods",
    "basic materials": "Materials",
    "energy": "Energy",
    "utilities": "Utilities",
}

UNMAPPED = "UNMAPPED"

# Sub-scale groups merge into their nearest neighbour (§14). A group of nine
# cannot carry a meaningful median or a sector rank, so the handbook folds it
# rather than leaving a rank computed off too few names.
#
# "Nearest" is an economic judgement, made once and written down here so the
# merge is reviewable rather than emergent. Each target is the group whose
# members face the most similar demand and capital cycle.
MERGE_TARGET = {
    "Commercial Services": "Capital Goods",
    "Aerospace & Defense": "Capital Goods",
    "Transport": "Capital Goods",
    "Banks": "Diversified Financials",
    "Telecom": "Media & Entertainment",
    "Autos": "Capital Goods",
    "Healthcare Equipment": "Healthcare Services",
    "Semiconductors": "Tech Hardware",
    "Insurance": "Diversified Financials",
    "Restaurants & Leisure": "Retail",
    "Consumer Staples": "Retail",
}


def resolve_merges(counts: dict, min_size: int = 15) -> dict:
    """Return {group: final group} after folding anything below min_size.

    Applied iteratively: a merge can push the target over the line, and a target
    that is itself sub-scale merges onward. Stops when nothing moves, so a
    cycle in MERGE_TARGET cannot hang the run.
    """
    mapping = {g: g for g in counts}
    sizes = dict(counts)
    for _ in range(len(counts) + 1):
        small = [g for g, n in sizes.items()
                 if n < min_size and g != UNMAPPED and MERGE_TARGET.get(g)]
        if not small:
            break
        g = min(small, key=lambda x: sizes[x])          # smallest first
        tgt = MERGE_TARGET[g]
        while mapping.get(tgt, tgt) != tgt:             # follow an existing merge
            tgt = mapping[tgt]
        if tgt == g:
            break
        for k, v in list(mapping.items()):
            if v == g:
                mapping[k] = tgt
        sizes[tgt] = sizes.get(tgt, 0) + sizes.pop(g)
    return mapping


def assign_group(sector: str | None, industry: str | None) -> str:
    """Exact industry match first; then the sector's default group; then
    UNMAPPED, which is counted and surfaced rather than quietly bucketed."""
    # a blank field arrives from pandas as NaN (a float, and truthy), so only a
    # real string is read; anything else is treated as missing
    ind = industry.strip().lower() if isinstance(industry, str) else ""
    if ind in _LOOKUP:
        return _LOOKUP[ind]
    for key, g in _LOOKUP.items():          # tolerate minor string drift
        if ind and (ind.startswith(key[:14]) or key.startswith(ind[:14])):
            return g
    sec = sector.strip().lower() if isinstance(sector, str) else ""
    return SECTOR_FALLBACK.get(sec, UNMAPPED)


def coverage_report(pairs) -> dict:
    """What share of a universe maps, and which strings fall through. The gate
    wants >=98% mapped and under 2% UNMAPPED."""
    from collections import Counter
    counts, missing = Counter(), Counter()
    for sector, industry in pairs:
        g = assign_group(sector, industry)
        counts[g] += 1
        if g == UNMAPPED:
            missing[industry] += 1
    total = sum(counts.values()) or 1
    return {"groups": dict(counts.most_common()),
            "mapped_pct": 100.0 * (total - counts[UNMAPPED]) / total,
            "unmapped_strings": dict(missing.most_common()),
            "below_min": {g: n for g, n in counts.items()
                          if g != UNMAPPED and n < 15}}
