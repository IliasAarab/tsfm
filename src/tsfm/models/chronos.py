import pandas as pd
import torch
from chronos import BaseChronosPipeline

from tsfm.data import infer_freq
from tsfm.exceptions import InvalidInputError
from tsfm.models.base import Model

BOLT_MODEL_ID = "amazon/chronos-bolt-small"


def prepare_data(df: pd.DataFrame, y: str, X: list[str] | None, ctx_len: int, oos_start: str, freq: str):
    cols = [y, *X] if X else [y]
    df = df[cols].copy()
    dfs = {}

    # Use appropriate offset based on frequency
    if freq.startswith("M"):
        off = pd.offsets.MonthEnd(1)
        ctx_off = pd.offsets.MonthEnd(ctx_len)
    elif freq.startswith("Q"):
        off = pd.offsets.QuarterEnd(1)
        ctx_off = pd.offsets.QuarterEnd(ctx_len)
    else:
        # Generic handling for other frequencies
        off = pd.tseries.frequencies.to_offset(freq)
        ctx_off = pd.tseries.frequencies.to_offset(f"{ctx_len}{freq}")

    for oos_date in df.index[df.index >= oos_start]:
        cutoff = oos_date - off
        dfs[cutoff.strftime("%Y-%m-%d")] = df.loc[oos_date - ctx_off : cutoff]

    return pd.concat([df.assign(cutoff=cutoff) for cutoff, df in dfs.items()]).reset_index(names=["oos_date"])


def make_fc_df(forecasts: pd.DataFrame, y_true: pd.DataFrame, horizon: int, freq: str) -> pd.DataFrame:
    y_true_col = y_true.columns[0]

    # Use appropriate offset based on frequency
    if freq.startswith("M"):
        off = pd.offsets.MonthEnd(horizon)
    elif freq.startswith("Q"):
        off = pd.offsets.QuarterEnd(horizon)
    else:
        off = pd.tseries.frequencies.to_offset(f"{horizon}{freq}")

    last_cutoff = y_true.index.max() - off
    quantile_cols = {str(q / 10): f"quantile_{q / 10}" for q in range(1, 10)}
    preds = (
        forecasts[forecasts["cutoff"] <= str(last_cutoff)]
        .rename(columns=quantile_cols | {"predictions": "y_pred"})
        .drop("target_name", axis=1)
        .sort_values(["cutoff", "oos_date"])
        .set_index(["cutoff", "oos_date"])
    )
    merged = (
        preds.reset_index(level="cutoff")  # keep 'cutoff' in df
        .merge(y_true, left_on="oos_date", right_index=True, how="left")
        .set_index("cutoff", append=True)  # restore MultiIndex order
        .reorder_levels(["cutoff", "oos_date"])
        .sort_index()
    )
    return merged[[y_true_col] + [col for col in merged if col != y_true_col]].rename(columns={y_true_col: "y_true"})


class Chronos(Model, name="chronos"):
    """Chronos-Bolt via BaseChronosPipeline, same external API as Chronos2."""

    @staticmethod
    def get_backbone() -> BaseChronosPipeline:
        return BaseChronosPipeline.from_pretrained(BOLT_MODEL_ID, device_map="auto")

    def _pred(
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
        mdl = self.get_backbone()

        # 1) Build rolling-window dataset (same as Chronos2)
        test_data = prepare_data(df, y, X, ctx_len, oos_start, freq)

        # 2) Prepare contexts as a list of 1D tensors, one per cutoff
        #    (avoids padding headaches, supported by Chronos/Chronos-Bolt)
        test_data = test_data.sort_values(["cutoff", "oos_date"])
        groups = test_data.groupby("cutoff", sort=True)

        contexts: list[torch.Tensor] = []
        cutoffs: list[str] = []

        for cutoff, g in groups:
            cutoffs.append(str(cutoff))
            series_vals = g[y].to_numpy()
            contexts.append(torch.tensor(series_vals, dtype=torch.float32))

        # 3) Call Chronos-Bolt to get quantiles
        quantile_levels = [q / 10 for q in range(1, 10)]

        # For Chronos-Bolt, predict_quantiles returns:
        #   quantiles: [num_series, prediction_length, num_quantiles]
        #   mean:      [num_series, prediction_length]
        quantiles, mean = mdl.predict_quantiles(
            inputs=contexts,
            prediction_length=horizon,
            quantile_levels=quantile_levels,
        )

        # 4) Build forecasts DataFrame in the same shape as Chronos2.predict_df
        records: list[dict] = []

        # Determine offset based on frequency
        if freq.startswith("M"):
            step_offset = lambda s: pd.offsets.MonthEnd(s + 1)
        elif freq.startswith("Q"):
            step_offset = lambda s: pd.offsets.QuarterEnd(s + 1)
        else:
            base_off = pd.tseries.frequencies.to_offset(freq)
            step_offset = lambda s: base_off * (s + 1)

        for i, cutoff_str in enumerate(cutoffs):
            cutoff_dt = pd.to_datetime(cutoff_str)

            for step in range(horizon):
                oos_date = cutoff_dt + step_offset(step)

                rec = {
                    "cutoff": cutoff_str,
                    "oos_date": oos_date,
                    "predictions": float(mean[i, step]),
                    "target_name": y,
                }

                for j, q in enumerate(quantile_levels):
                    rec[str(q)] = float(quantiles[i, step, j])

                records.append(rec)

        forecasts = pd.DataFrame.from_records(records)

        # 5)  Get the final eval DataFrame
        return make_fc_df(forecasts, df[[y]], horizon, freq)
