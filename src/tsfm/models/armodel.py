from typing import cast

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from tsfm.exceptions import InvalidInputError
from tsfm.models.base import Model


def create_lags(var: pd.Series, lags: int = 1) -> pd.DataFrame:
    """Return DataFrame with columns var_lag0..var_lag{lags} aligned to var.index."""
    return pd.DataFrame(
        {f"{var.name}_lag{i}": var.shift(i) for i in range(lags + 1)}, index=var.index
    )


def get_design_matrix(
    ys: pd.Series, y_lags: int = 1, horizon: int = 1
) -> tuple[pd.Series, pd.DataFrame]:
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


class ARModel(Model, name="armodel"):
    def pred(
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

        y_lags = ctx_len - 1
        # return pred(df[y], y_lags, horizon, oos_start)

        parts = []
        for h in range(1, horizon + 1):
            # yhat: Series indexed by oos_date (t+h), values = predictions
            yhat = pred(ys=df[y], y_lags=y_lags, horizon=h, oos_start=oos_start)

            df_h = pd.DataFrame(
                {
                    "y_pred": yhat,
                    "y_true": df[y].reindex(yhat.index),
                }
            )

            cutoff_idx = df_h.index - pd.DateOffset(months=h)  # type: ignore
            df_h.index = pd.MultiIndex.from_arrays(
                [cutoff_idx, df_h.index], names=["cutoff", "oos_date"]
            )
            parts.append(df_h)
        out = pd.concat(parts).sort_index()
        return out
