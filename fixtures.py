"""
fixtures.py
-----------
Reproducible synthetic datasets for testing the Artefact Registry.
No external dependencies. Fixed seed for determinism.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def _date_index(n: int, start: str = "2020-01-02", freq: str = "B") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def make_returns_series(n: int = 1000, name: str = "returns") -> pd.Series:
    """
    Daily return series with realistic vol clustering (GARCH-like).
    Values roughly in [-0.05, +0.05].
    """
    rng = np.random.default_rng(42)
    vol = np.ones(n) * 0.01
    rets = np.zeros(n)
    for i in range(1, n):
        vol[i] = np.sqrt(0.00001 + 0.85 * vol[i-1]**2 + 0.10 * rets[i-1]**2)
        rets[i] = vol[i] * rng.standard_normal()
    return pd.Series(rets, index=_date_index(n), name=name, dtype=float)


def make_position_history(
    n_dates: int = 500,
    n_assets: int = 50,
) -> pd.DataFrame:
    """
    Weight matrix. Each row sums to 1.0. Columns are asset IDs.
    Simulates a concentrated portfolio that gradually diversifies.
    """
    rng = np.random.default_rng(43)
    asset_ids = [f"ASSET_{i:03d}" for i in range(n_assets)]
    raw = rng.dirichlet(np.ones(n_assets) * 0.5, size=n_dates)
    # Zero out small weights to simulate index construction thresholds
    raw[raw < 0.005] = 0.0
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    weights = raw / row_sums
    return pd.DataFrame(weights, index=_date_index(n_dates), columns=asset_ids)


def make_factor_exposures(
    n_dates: int = 500,
    n_factors: int = 5,
) -> pd.DataFrame:
    """
    Factor loading matrix. Values roughly mean-zero, unit-variance.
    Factor names: Value, Momentum, Quality, LowVol, Size.
    """
    rng = np.random.default_rng(44)
    factor_names = ["Value", "Momentum", "Quality", "LowVol", "Size"][:n_factors]
    data = rng.standard_normal((n_dates, n_factors))
    # Add a slow drift to make the series interesting
    drift = np.linspace(-0.5, 0.5, n_dates)
    data[:, 0] += drift  # Value factor drifts
    return pd.DataFrame(data, index=_date_index(n_dates), columns=factor_names)


def make_external_benchmark(n: int = 1000, name: str = "benchmark_returns") -> pd.Series:
    """
    Independent return series for join/comparison tests.
    Correlated ~0.6 with a standard normal, so partially related to returns.
    """
    rng = np.random.default_rng(45)
    base = rng.standard_normal(n) * 0.008
    noise = rng.standard_normal(n) * 0.005
    rets = 0.6 * base + 0.4 * noise
    return pd.Series(rets, index=_date_index(n), name=name, dtype=float)


def make_universe_file() -> pd.DataFrame:
    """
    Universe file with ISIN-like IDs and numeric attributes.
    Suitable for file upload ingestion tests.
    """
    rng = np.random.default_rng(46)
    n = 200
    isins = [f"GB{i:010d}" for i in range(n)]
    sectors = rng.choice(
        ["Technology", "Financials", "Healthcare", "Energy", "Industrials"],
        size=n,
    )
    market_cap = rng.lognormal(mean=8, sigma=1.5, size=n)
    adtv = rng.lognormal(mean=4, sigma=1.2, size=n)
    float_factor = rng.uniform(0.3, 1.0, size=n)

    return pd.DataFrame({
        "isin": isins,
        "sector": sectors,
        "market_cap_usd_m": market_cap.round(1),
        "adtv_usd_m": adtv.round(2),
        "float_factor": float_factor.round(3),
    })


def make_universe_csv_bytes() -> bytes:
    """Returns universe file as CSV bytes, for file upload ingestion tests."""
    df = make_universe_file()
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def make_returns_csv_bytes() -> bytes:
    """Returns return series as CSV bytes with a date column."""
    s = make_returns_series(500)
    df = s.reset_index()
    df.columns = ["date", "return"]
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def make_all_fixtures() -> dict:
    """
    Returns a dict of all standard fixtures.
    Suitable for bulk registration into an ArtefactRegistry.
    """
    return {
        "returns":          make_returns_series(1000),
        "position_history": make_position_history(500, 50),
        "factor_exposures": make_factor_exposures(500, 5),
        "benchmark":        make_external_benchmark(1000),
    }
