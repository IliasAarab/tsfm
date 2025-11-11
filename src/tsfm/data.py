import warnings

import numpy as np
import pandas as pd


def infer_freq(df: pd.DataFrame) -> str:
    """Infer frequency from DataFrame index.

    Returns pandas frequency string like 'M', 'Q', 'W', 'D', etc.
    Falls back to 'M' if inference fails.

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        Frequency string (e.g., 'M' for monthly, 'Q' for quarterly).

    Raises:
        TypeError: If DataFrame index is not a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        msg = "DataFrame index must be DatetimeIndex"
        raise TypeError(msg)

    freq = pd.infer_freq(df.index)
    if freq is None:
        # Try to get from index.freq attribute
        freq = df.index.freq
        if freq is None:
            warnings.warn("Could not infer frequency, defaulting to 'ME' (monthly)", stacklevel=2)
            return "ME"
        freq = freq.freqstr

    return freq


def generator(
    n=100,
    phi=0.7,
    beta0=0.5,
    beta1=0.3,
    sigma=1.0,
    trend=0.01,
    season_period=12,
    season_ampl=1.0,
    seed=0,
):
    """
    Simulate time series data:
        y_t = phi * y_{t-1} + beta0 * x_t + beta1 * x_{t-1} + trend * t
              + season_ampl * sin(2πt/season_period) + eps_t

    Args:
        n (int): Number of time points.
        phi (float): AR(1) coefficient for y.
        beta0 (float): Coefficient for current x.
        beta1 (float): Coefficient for previous x.
        sigma (float): Stddev of noise.
        trend (float): Linear trend coefficient.
        season_period (int): Period of the seasonality.
        season_ampl (float): Amplitude of the seasonality.
        seed (int): Random seed.

    Returns:
        pd.DataFrame with columns ['x', 'y']
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n + 1)  # n+1 for lag
    y = np.zeros(n + 1)
    eps = rng.normal(scale=sigma, size=n + 1)

    t_idx = np.arange(1, n + 1)
    season = season_ampl * np.sin(2 * np.pi * t_idx / season_period)
    tr = trend * t_idx

    for t in range(1, n + 1):
        y[t] = phi * y[t - 1] + beta0 * x[t] + beta1 * x[t - 1] + tr[t - 1] + season[t - 1] + eps[t]

    df = pd.DataFrame({"x": x[1:], "y": y[1:]})
    df.index = pd.date_range("2000-01", periods=len(df), freq="M")
    return df


def split_is_oos(df, test_frac=0.1):
    """
    Split a DataFrame into in-sample and out-of-sample parts.

    Args:
        df (pd.DataFrame): Data to split.
        test_frac (float): Fraction of data to use as out-of-sample.

    Returns:
        df_is (pd.DataFrame): In-sample DataFrame.
        df_oos (pd.DataFrame): Out-of-sample DataFrame.
    """
    n = len(df)
    split_idx = int(np.floor((1 - test_frac) * n))
    df_is = df.iloc[:split_idx].copy()
    df_oos = df.iloc[split_idx:].copy()
    return df_is, df_oos
