"""
fixtures.py
-----------
Reproducible synthetic datasets for testing the Artefact Registry.
No external dependencies. Fixed seed for determinism.

All datasets share consistent instrument_id / ISIN / SEDOL / CUSIP identifiers.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd

N_INSTRUMENTS = 1000
_START_DATE = "2020-01-02"
_END_DATE = "2025-01-01"


# ---------------------------------------------------------------------------
# Shared identifiers (deterministic, aligned across all fixtures)
# ---------------------------------------------------------------------------

def _instrument_ids() -> list[str]:
    return [f"INST_{i:04d}" for i in range(N_INSTRUMENTS)]


def _business_dates() -> pd.DatetimeIndex:
    return pd.bdate_range(start=_START_DATE, end=_END_DATE)


def _effective_dates() -> pd.DatetimeIndex:
    """First business day of each month — monthly rebalancing cadence."""
    return pd.bdate_range(start=_START_DATE, end=_END_DATE, freq="BMS")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_universe_file() -> pd.DataFrame:
    """
    Static universe descriptor table.

    Index : instrument_id
    Columns : stock_name, region, industry, ISIN, freefloat, SEDOL, CUSIP

    Identifiers (instrument_id, ISIN, SEDOL, CUSIP) are consistent with every
    other fixture in this module.
    """
    rng = np.random.default_rng(46)
    n = N_INSTRUMENTS
    ids = _instrument_ids()

    regions = rng.choice(
        ["North America", "Europe", "Asia Pacific", "Emerging Markets"],
        size=n,
        p=[0.40, 0.30, 0.20, 0.10],
    )
    industries = rng.choice(
        [
            "Technology", "Financials", "Healthcare", "Energy", "Industrials",
            "Consumer Discretionary", "Consumer Staples", "Utilities",
            "Materials", "Real Estate",
        ],
        size=n,
    )
    freefloat = rng.uniform(0.30, 1.00, size=n).round(3)

    return pd.DataFrame(
        {
            "stock_name": [f"Company {i:04d} Ltd" for i in range(n)],
            "region":     regions,
            "industry":   industries,
            "ISIN":       [f"GB{i:010d}" for i in range(n)],
            "freefloat":  freefloat,
            "SEDOL":      [f"B{i:06d}" for i in range(n)],
            "CUSIP":      [f"{i:09d}" for i in range(n)],
        },
        index=pd.Index(ids, name="instrument_id"),
    )


def make_stock_returns_series() -> pd.DataFrame:
    """
    Daily total return matrix with realistic vol clustering (GARCH-like).

    Index   : date — 5 years of business days (~1 305 rows); DAILY frequency.
    Columns : instrument_id — 1 000 stocks (shared key with all other fixtures)

    Values represent daily total return  r_t = p_t / p_{t-1} − 1, where the
    common market factor follows GARCH(1,1) dynamics and each stock adds
    idiosyncratic noise scaled by a random beta.

    ALIGNMENT: date (daily) is at a different frequency than effective_date
    (monthly) used by position_history and factor_exposures.  Forward-fill or
    asof-join on the time axis before concatenating with those artefacts.
    """
    rng = np.random.default_rng(42)
    dates = _business_dates()
    n_dates = len(dates)
    ids = _instrument_ids()
    n = N_INSTRUMENTS

    # Market factor — GARCH(1,1)
    market_vol = np.empty(n_dates)
    market_ret = np.empty(n_dates)
    market_vol[0] = 0.010
    market_ret[0] = 0.0
    for t in range(1, n_dates):
        market_vol[t] = np.sqrt(
            max(1e-8, 1e-5 + 0.85 * market_vol[t - 1] ** 2 + 0.10 * market_ret[t - 1] ** 2)
        )
        market_ret[t] = market_vol[t] * rng.standard_normal()

    # Idiosyncratic component (vectorised across stocks)
    betas = rng.uniform(0.5, 1.5, size=n)
    idio = rng.standard_normal((n_dates, n)) * 0.012
    rets = market_ret[:, None] * betas[None, :] + idio

    return pd.DataFrame(
        rets,
        index=pd.DatetimeIndex(dates, name="date"),
        columns=pd.Index(ids, name="instrument_id"),
    )


def make_position_history() -> pd.DataFrame:
    """
    Portfolio weight history at monthly rebalancing dates.

    Index   : (effective_date, instrument_id) — MultiIndex; MONTHLY frequency.
              effective_date = first business day of each month (~61 dates).
              instrument_id is a shared key with all other fixtures.
    Columns : benchmark_weight, index_weight, mcap_usd

    benchmark_weight and index_weight each sum to exactly 1.0 for every
    effective_date. mcap_usd is in USD millions and varies across dates.

    ALIGNMENT: effective_date (monthly) differs from date (daily) in
    stock_returns.  Forward-fill weights to business-day frequency or use an
    asof-join before combining with that artefact.
    Can be merged directly with factor_exposures on (effective_date, instrument_id).
    """
    rng = np.random.default_rng(43)
    eff_dates = _effective_dates()
    ids = _instrument_ids()
    n_dates = len(eff_dates)
    n = N_INSTRUMENTS

    # Dirichlet weights — sparse (small weights zeroed out and renormalised)
    bm = rng.dirichlet(np.ones(n) * 0.5, size=n_dates)
    ix = rng.dirichlet(np.ones(n) * 0.3, size=n_dates)

    bm[bm < 2e-4] = 0.0
    ix[ix < 2e-4] = 0.0
    bm /= bm.sum(axis=1, keepdims=True)
    ix /= ix.sum(axis=1, keepdims=True)

    # Market cap: log-normal base with small random drift per date
    base_mcap = rng.lognormal(mean=10.0, sigma=1.5, size=n)
    mcap_noise = rng.lognormal(mean=0.0, sigma=0.05, size=(n_dates, n))
    mcap = base_mcap[None, :] * mcap_noise  # (n_dates, n)

    midx = pd.MultiIndex.from_arrays(
        [np.repeat(eff_dates, n), np.tile(ids, n_dates)],
        names=["effective_date", "instrument_id"],
    )
    return pd.DataFrame(
        {
            "benchmark_weight": bm.ravel(),
            "index_weight":     ix.ravel(),
            "mcap_usd":         mcap.ravel(),
        },
        index=midx,
    )


def make_factor_exposures() -> pd.DataFrame:
    """
    Factor loading matrix at monthly rebalancing dates.

    Index   : (effective_date, instrument_id) — MultiIndex; MONTHLY frequency.
              effective_date = first business day of each month, aligned with
              position_history.  instrument_id is a shared key with all fixtures.
    Columns : factor — "Value", "Quality", "Momentum", "Low Vol", "Size"

    Loadings are approximately mean-zero and unit-variance cross-sectionally.
    The Value factor carries a slow time-series drift to make the data
    non-trivially time-varying.

    ALIGNMENT: shares (effective_date, instrument_id) with position_history —
    merge directly on those keys.  To combine with stock_returns (daily date),
    forward-fill factor exposures to business-day frequency first.
    """
    rng = np.random.default_rng(44)
    eff_dates = _effective_dates()
    ids = _instrument_ids()
    n_dates = len(eff_dates)
    n = N_INSTRUMENTS
    factor_names = ["Value", "Quality", "Momentum", "Low Vol", "Size"]
    n_factors = len(factor_names)

    data = rng.standard_normal((n_dates, n, n_factors))
    drift = np.linspace(-0.5, 0.5, n_dates)
    data[:, :, 0] += drift[:, None]  # Value factor drifts over time

    midx = pd.MultiIndex.from_arrays(
        [np.repeat(eff_dates, n), np.tile(ids, n_dates)],
        names=["effective_date", "instrument_id"],
    )
    return pd.DataFrame(
        data.reshape(-1, n_factors),
        index=midx,
        columns=pd.Index(factor_names, name="factor_id"),
    )


def make_user_signals() -> pd.DataFrame:
    """
    Proprietary signal matrix at weekly (Wednesday) signal dates.

    Index   : (ISIN, signal_date) — MultiIndex; WEEKLY frequency (every Wednesday).
              signal_date is intentionally misaligned with both effective_date
              (monthly) and date (daily).  ISINs are consistent with
              make_universe_file() and all other fixtures.
    Columns : "Proprietary 1", "Proprietary 2" (AR(1) persistence across dates)

    ALIGNMENT: signal_date (weekly) aligns with neither date (daily) nor
    effective_date (monthly).  Join to other artefacts via ISIN using an
    asof-join or forward-fill on the time axis.
    """
    rng = np.random.default_rng(47)
    universe = make_universe_file()
    isins = universe["ISIN"].tolist()

    signal_dates = pd.bdate_range(start=_START_DATE, end=_END_DATE, freq="W-WED")
    n_signals = len(signal_dates)
    n_isins = len(isins)

    raw = rng.standard_normal((n_signals, n_isins, 2))
    # AR(1) persistence across signal dates
    for t in range(1, n_signals):
        raw[t] = 0.70 * raw[t - 1] + 0.30 * raw[t]

    midx = pd.MultiIndex.from_arrays(
        [np.tile(isins, n_signals), np.repeat(signal_dates, n_isins)],
        names=["ISIN", "signal_date"],
    )
    return pd.DataFrame(
        raw.reshape(-1, 2),
        index=midx,
        columns=["Proprietary 1", "Proprietary 2"],
    )


# ---------------------------------------------------------------------------
# Serialisers
# ---------------------------------------------------------------------------

def make_universe_csv_bytes() -> bytes:
    """Universe file as CSV bytes, for file-upload ingestion tests."""
    buf = io.BytesIO()
    make_universe_file().to_csv(buf)
    return buf.getvalue()


def make_returns_csv_bytes() -> bytes:
    """Stock returns matrix as CSV bytes (date index + instrument_id columns)."""
    buf = io.BytesIO()
    make_stock_returns_series().to_csv(buf)
    return buf.getvalue()


def make_all_fixtures() -> dict:
    """
    Returns all standard fixtures keyed by name, ready for bulk registration
    into an ArtefactRegistry.
    """
    return {
        "universe":         make_universe_file(),
        "stock_returns":    make_stock_returns_series(),
        "position_history": make_position_history(),
        "factor_exposures": make_factor_exposures(),
        "user_signals":     make_user_signals(),
    }
