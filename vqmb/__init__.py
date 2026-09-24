"""VQMB - a quantitative stock-ranking model: Value, Quality, business
Momentum (B) and price Momentum (M).

    import vqmb
    df = vqmb.run(["AAPL", "MSFT", "JPM"], api_key="YOUR_FMP_KEY")

Returns one pandas DataFrame, one row per scored name, with every source
figure, metric, percentile, pillar, rank, gate, flag and grade the model
produces. See README.md for the column reference.
"""
from . import config, darvas, facts, fmp, groups, metrics, ranking, workings  # noqa: F401
from .engine import (VQMBError, fetch_benchmark, fetch_data, index_constituents,  # noqa: F401
                     run, run_from_data, score)

__all__ = ["run", "run_from_data", "fetch_data", "fetch_benchmark", "index_constituents",
           "score", "VQMBError"]
