from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def get_horizon_groupby(df: pd.DataFrame, freq: str | None = None) -> pd.Index:
    """Calculate horizon index based on frequency.

    For monthly: 1=next month, 2=two months ahead, ...
    For quarterly: 1=next quarter, 2=two quarters ahead, ...

    Args:
        df: DataFrame with MultiIndex (cutoff, oos_date).
        freq: Frequency string (e.g., 'M', 'Q'). If None, infers from index.

    Returns:
        Index of horizon values.
    """
    cut = pd.DatetimeIndex(df.index.get_level_values(0))
    oos = pd.DatetimeIndex(df.index.get_level_values(1))

    if freq is None:
        # Try to infer from the index values
        freq = pd.infer_freq(cut)
        if freq is None:
            freq = "M"  # fallback to monthly

    cut_ord = cut.to_period(freq).astype("int64")
    oos_ord = oos.to_period(freq).astype("int64")
    return oos_ord - cut_ord


@dataclass
class ForecastOutput:
    df_preds: pd.DataFrame = field(repr=False)  # MultiIndex[cutoﬀ, oos_date], cols: y_true,y_pred
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- metrics -------------------------------------------------------------
    def _agg_mean(self, s: pd.Series, name: str, post=None) -> pd.DataFrame:
        freq = self.meta.get("freq")
        g = get_horizon_groupby(self.df_preds, freq=freq)
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

    def _summary(self, digits: int = 4) -> str:
        """Return a statsmodels-like text summary with metrics side by side."""
        idx = self.df_preds.index
        cut = pd.DatetimeIndex(idx.get_level_values(0))
        oos = pd.DatetimeIndex(idx.get_level_values(1))
        freq = self.meta.get("freq")
        horizons = get_horizon_groupby(self.df_preds, freq=freq)
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
        Line plot of y_true vs y_pred for a given horizon.
        Adds predictive intervals if quantile_* columns exist.

        Args:
            horizon: Forecast horizon (1=next period, 2=two periods ahead, etc.)
            ax: Optional matplotlib Axes to plot on
            start: Optional start date for plot range
            end: Optional end date for plot range
            return_ax: If True, return the Axes object

        Returns:
            Axes object if return_ax=True, else None
        """
        freq = self.meta.get("freq")
        g = get_horizon_groupby(self.df_preds, freq=freq)
        m = g == horizon
        if not m.any():
            msg = f"No data for horizon={horizon}."
            raise ValueError(msg)

        # MultiIndex (cutoff, oos_date); use oos_date as x
        sub = self.df_preds.loc[m].copy()
        oos = sub.index.get_level_values(1)
        sub.index = pd.DatetimeIndex(oos, name="oos_date")
        sub.sort_index(inplace=True)

        if start is not None or end is not None:
            sub = sub.loc[start:end]

        ax = ax or plt.gca()

        # --- Quantile fan ---
        qcols = [c for c in sub.columns if c.startswith("quantile_")]

        def q_level(col: str) -> float:
            return float(col.split("_", 1)[1])

        # sort to have deterministic order
        qcols = sorted(qcols, key=q_level)
        qvals = np.array([q_level(c) for c in qcols])

        def nearest_pair(target_lo: float, target_hi: float):
            lo_idx = np.argmin(np.abs(qvals - target_lo))
            hi_idx = np.argmin(np.abs(qvals - target_hi))
            lo_col = qcols[lo_idx]
            hi_col = qcols[hi_idx]
            actual_coverage = qvals[hi_idx] - qvals[lo_idx]
            return lo_col, hi_col, actual_coverage

        # wider band first, then narrower
        for target_lo, target_hi, nominal_label, alpha in [
            (0.10, 0.90, "80% PI", 0.2),
            (0.20, 0.8, "60% PI", 0.4),
        ]:
            q_lo, q_hi, actual_cov = nearest_pair(target_lo, target_hi)
            ax.fill_between(
                sub.index,
                sub[q_lo].astype(float),
                sub[q_hi].astype(float),
                alpha=alpha,
                label=nominal_label,
            )

        # y_true line and point forecasts
        ax.plot(
            sub.index,
            sub["y_true"],
            lw=2,
            alpha=0.9,
            label="y_true",
            c="k",
            ls="--",
        )
        ax.scatter(
            sub.index,
            sub["y_pred"],
            alpha=0.9,
            label="y_pred",
            c="firebrick",
        )

        ax.set_xlabel("OOS date")
        ax.set_ylabel("Prediction")
        ttl = self.meta.get("model") or "Forecast"
        ax.set_title(f"{ttl} - y_true vs y_pred @ horizon={horizon}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.figure.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        return ax if return_ax else None
