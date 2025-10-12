from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def get_horizon_groupby(df: pd.DataFrame) -> pd.Index:
    """1-based monthly horizon: 1=next month, 2=two months ahead, ..."""
    cut = pd.DatetimeIndex(df.index.get_level_values(0))
    oos = pd.DatetimeIndex(df.index.get_level_values(1))
    cut_ord = cut.to_period("M").astype("int64")
    oos_ord = oos.to_period("M").astype("int64")
    return oos_ord - cut_ord


@dataclass
class ForecastOutput:
    df_preds: pd.DataFrame = field(repr=False)  # MultiIndex[cutoﬀ, oos_date], cols: y_true,y_pred
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- metrics -------------------------------------------------------------
    def _agg_mean(self, s: pd.Series, name: str, post=None) -> pd.DataFrame:
        g = get_horizon_groupby(self.df_preds)
        out = s.groupby(g).mean()
        if post is not None:
            out = post(out)
        out.index.name = "horizon"
        df = out.to_frame(name=name)
        df.loc["average", name] = df[name].mean()
        return df

    @cached_property
    def rmsfe(self) -> pd.DataFrame:
        se = (self.df_preds["y_true"] - self.df_preds["y_pred"]) ** 2
        return self._agg_mean(se, name="rmsfe", post=np.sqrt)

    @cached_property
    def mae(self) -> pd.DataFrame:
        ae = (self.df_preds["y_true"] - self.df_preds["y_pred"]).abs()
        return self._agg_mean(ae, name="mae")

    @cached_property
    def me(self) -> pd.DataFrame:
        err = self.df_preds["y_true"] - self.df_preds["y_pred"]
        return self._agg_mean(err, name="me")

    def metric(self, name: Literal["rmsfe", "mae", "me"]) -> pd.DataFrame:
        return getattr(self, name)

    # ---- summary -------------------------------------------------------------
    def _summary(self, digits: int = 4) -> str:
        """Return a statsmodels-like text summary with metrics side by side."""
        idx = self.df_preds.index
        cut = pd.DatetimeIndex(idx.get_level_values(0))
        oos = pd.DatetimeIndex(idx.get_level_values(1))
        horizons = get_horizon_groupby(self.df_preds)
        n_obs = len(self.df_preds)

        # combine metrics into one DataFrame
        df = pd.concat([self.rmsfe, self.mae, self.me], axis=1)
        df.index.name = "horizon"

        float_fmt = f"%.{digits}f"
        metrics_txt = df.to_string(float_format=float_fmt)

        # flatten meta for printing
        meta_items = []
        for k, v in (self.meta or {}).items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                meta_items.append(f"{k}={v}")
        meta_line = " | ".join(meta_items)

        lines = [
            "================== Forecast Results Summary ==================",
            f"Observations: {n_obs}",
            f"Cutoff range : {cut.min().date()}  -  {cut.max().date()}",
            f"OOS range    : {oos.min().date()}  -  {oos.max().date()}",
            f"Horizons     : {', '.join(map(str, np.sort(np.unique(horizons))))}",
            f"Meta         : {meta_line}" if meta_line else "Meta         : (none)",
            "--------------------------------------------------------------",
            metrics_txt,
            "==============================================================",
        ]
        return "\n".join(lines)

    def summary(self, digits: int = 4) -> None:
        print(self._summary(digits))  # noqa: T201

    def plot_actual_vs_pred(
        self,
        horizon: int,
        ax: Axes | None = None,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        return_ax: bool = False,
    ) -> Axes | None:
        """
        Line plot of y_true vs y_pred for a given horizon (months).
        Adds 50% and 80% predictive intervals if quantile_* columns exist.
        """
        g = get_horizon_groupby(self.df_preds)
        m = g == horizon
        if not m.any():
            raise ValueError(f"No data for {horizon=}.")

        sub = self.df_preds.loc[m].copy()
        sub.index = pd.DatetimeIndex(sub.index.get_level_values(1), name="oos_date")
        sub.sort_index(inplace=True)
        if start is not None or end is not None:
            sub = sub.loc[start:end]

        ax = ax or plt.gca()

        # --- Quantile fan: 80% then 50% (plot wider first, then narrower) ---
        qcols = [c for c in sub.columns if c.startswith("quantile_")]
        if qcols:

            def q_level(col: str) -> float:
                # "quantile_0.1" -> 0.1
                return float(col.split("_", 1)[1])

            def closest_quantile(target: float) -> str:
                return min(qcols, key=lambda c: abs(q_level(c) - target))

            bands = [
                (0.10, 0.90, "80% PI", 0.99),
                (0.25, 0.75, "50% PI", 0.99),
            ]
            for lo_tgt, hi_tgt, label, alpha in bands:
                q_lo = closest_quantile(lo_tgt)
                q_hi = closest_quantile(hi_tgt)
                ax.fill_between(
                    sub.index,
                    sub[q_lo].astype(float),
                    sub[q_hi].astype(float),
                    alpha=alpha,
                    label=label,
                    linewidth=3,
                )

        # y_true and point preds
        ax.plot(sub["y_true"], lw=2, alpha=0.9, label="y_true", c="k", ls="--")
        ax.scatter(sub.index, sub["y_pred"], lw=2, alpha=0.9, label="y_pred", c="firebrick")

        ax.set_xlabel("OOS date")
        ax.set_ylabel("Value")
        ttl = self.meta.get("model_name") or "Forecast"
        ax.set_title(f"{ttl} - y_true vs y_pred @ horizon={horizon}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.figure.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        return ax if return_ax else None
