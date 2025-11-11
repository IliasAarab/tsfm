from typing import cast

import pandas as pd
import statsmodels.api as sm
from pandas import IndexSlice as Idx
from pandas.tseries.frequencies import to_offset
from statsmodels.regression.linear_model import RegressionResultsWrapper

from tsfm.data import infer_freq
from tsfm.exceptions import InvalidInputError
from tsfm.models.base import Model


def create_lags(var: pd.Series, lags: int = 1) -> pd.DataFrame:
    """Return DataFrame with columns var_lag0..var_lag{lags} aligned to var.index."""
    return pd.DataFrame({f"{var.name}_lag{i}": var.shift(i) for i in range(lags + 1)}, index=var.index)


def get_design_matrix(ys: pd.Series, y_lags: int = 1, horizon: int = 1) -> tuple[pd.Series, pd.DataFrame]:
    df_y = create_lags(var=ys, lags=y_lags)
    X = sm.add_constant(df_y)
    X = cast(pd.DataFrame, X)
    ys = ys.copy().shift(-horizon)
    mask = X.notna().all(axis=1) & ys.notna()
    return ys[mask], X[mask]


def fit(ys: pd.Series, y_lags: int = 1, horizon: int = 1) -> RegressionResultsWrapper:
    ys, X = get_design_matrix(ys, y_lags, horizon)
    return sm.OLS(ys, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})


def pred(
    ys: pd.Series,
    y_lags: int = 1,
    horizon: int = 1,
    oos_start: str = "2020-01-31",
):
    yhs = []
    cutoffs = ys.index[ys.index >= oos_start]

    for cutoff in cutoffs:
        # Training
        tr_y = ys[ys.index < cutoff]
        mdl = fit(ys=tr_y, y_lags=y_lags, horizon=horizon)
        # Inference
        te_y = ys[ys.index <= cutoff]
        te_y, te_x = get_design_matrix(te_y, y_lags=y_lags, horizon=horizon)
        te_y, te_x = te_y.iloc[-1:], te_x.iloc[-1:]
        yh = mdl.predict(te_x)
        yh.index = [cutoff]
        yhs.append(yh)
    return pd.concat(yhs)


def _to_period_end(idx: pd.Index, freq: str) -> pd.DatetimeIndex:
    """Align datetime index to period end based on frequency."""
    idx = pd.DatetimeIndex(idx)
    return idx.to_period(freq).to_timestamp(freq)


def build_oos_panel(ys: pd.Series, *, y_lags: int, horizon: int, oos_start: str, freq: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for h in range(1, horizon + 1):
        yhat = pred(ys=ys, y_lags=y_lags, horizon=h, oos_start=oos_start)  # index = oos_date
        oos_idx = _to_period_end(yhat.index, freq)
        cutoff_idx = (oos_idx.to_period(freq) - h).to_timestamp(freq)  # info time (period-end aligned)

        df_h = pd.DataFrame(
            {"y_true": ys.reindex(oos_idx).to_numpy(), "y_pred": yhat.to_numpy()},
            index=pd.MultiIndex.from_arrays([cutoff_idx, oos_idx], names=["cutoff", "oos_date"]),
        )
        parts.append(df_h)

    out = pd.concat(parts).sort_index()
    start = pd.to_datetime(oos_start) - to_offset(freq)
    end = ys.index[-1] - to_offset(freq) * horizon
    out = out[Idx[start:end]]
    return out[["y_true", "y_pred"]]


class ARModel(Model, name="armodel"):
    def get_backbone(self):  # noqa: PLR6301
        return None

    def _pred(  # noqa: PLR6301
        self,
        df: pd.DataFrame,
        y: str,
        X: list[str] | None = None,
        ctx_len: int = 1,
        horizon: int = 1,
        oos_start: str = "2020-01-31",
    ) -> pd.DataFrame:
        if X:
            msg = "Ilias: No covariates supported for this model!"
            raise InvalidInputError(msg)
        freq = infer_freq(df)
        y_lags = ctx_len - 1
        return build_oos_panel(ys=df[y], y_lags=y_lags, horizon=horizon, oos_start=oos_start, freq=freq)
