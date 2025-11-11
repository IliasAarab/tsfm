import pandas as pd
from chronos import BaseChronosPipeline, Chronos2Pipeline

from tsfm.data import infer_freq
from tsfm.models.base import Model

MODEL_ID = "amazon/chronos-2"


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


class Chronos2(Model, name="chronos2"):
    @staticmethod
    def get_backbone() -> Chronos2Pipeline:
        return BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="auto")

    def _pred(
        self,
        df: pd.DataFrame,
        y: str,
        X: list[str] | None = None,
        ctx_len: int = 1,
        horizon: int = 1,
        oos_start: str = "2020-01-31",
    ) -> pd.DataFrame:
        freq = infer_freq(df)
        mdl = self.get_backbone()
        test_data = prepare_data(df, y, X, ctx_len, oos_start, freq)
        forecasts = mdl.predict_df(
            test_data,
            id_column="cutoff",
            timestamp_column="oos_date",
            target=y,
            prediction_length=horizon,
            batch_size=256,
        )
        return make_fc_df(forecasts, df[[y]], horizon, freq)
