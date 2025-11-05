import pandas as pd
from chronos import BaseChronosPipeline, Chronos2Pipeline

from tsfm.models.base import Model

MODEL_ID = "amazon/chronos-2"


def prepare_data(df: pd.DataFrame, y: str, X: list[str] | None, ctx_len: int, oos_start: str):
    cols = [y, *X] if X else [y]
    df = df[cols].copy()
    dfs = {}
    for oos_date in df.index[df.index >= oos_start]:
        cutoff = oos_date - pd.offsets.MonthEnd(1)
        dfs[cutoff.strftime("%Y-%m-%d")] = df.loc[oos_date - pd.offsets.MonthEnd(ctx_len) : cutoff]

    return pd.concat([df.assign(cutoff=cutoff) for cutoff, df in dfs.items()]).reset_index(names=["oos_date"])


def make_fc_df(forecasts: pd.DataFrame, y_true: pd.DataFrame, horizon: int) -> pd.DataFrame:
    y_true_col = y_true.columns[0]
    last_cutoff = y_true.index.max() - pd.offsets.MonthEnd(horizon)
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
        return BaseChronosPipeline.from_pretrained(MODEL_ID)

    def _pred(
        self,
        df: pd.DataFrame,
        y: str,
        X: list[str] | None = None,
        ctx_len: int = 1,
        horizon: int = 1,
        oos_start: str = "2020-01-31",
    ) -> pd.DataFrame:
        mdl = self.get_backbone()
        test_data = prepare_data(df, y, X, ctx_len, oos_start)
        forecasts = mdl.predict_df(
            test_data,
            id_column="cutoff",
            timestamp_column="oos_date",
            target=y,
            prediction_length=horizon,
            batch_size=256,
        )
        return make_fc_df(forecasts, df[[y]], horizon)
