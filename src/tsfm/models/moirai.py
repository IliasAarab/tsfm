import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import TestData, split
from pandas.tseries.frequencies import to_offset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

from tsfm.data import infer_freq
from tsfm.models.base import Model

MODEL_ID = "Salesforce/moirai-1.1-R-small"


def make_fc_df(forecasts, y_true_series: pd.Series, freq: str) -> pd.DataFrame:
    out = []
    for fc in forecasts:
        off = to_offset(freq)

        # start date -> Timestamp (align period to period-end)
        start = fc.start_date.to_timestamp(how="end")

        horizon = len(fc.mean)
        oos_idx = pd.date_range(start=start, periods=horizon, freq=off)

        # Normalize timestamps to remove time component for proper reindexing
        oos_idx = pd.DatetimeIndex(oos_idx.normalize())
        cutoff = pd.Timestamp(start.normalize()) - off

        y_pred = fc.quantile(0.5)  # median = point forecast
        quantiles = {f"quantile_{q}": fc.quantile(q) for q in [i / 10 for i in range(1, 10)]}
        y_true = y_true_series.reindex(oos_idx).to_numpy()

        out.append(
            pd.DataFrame({"cutoff": cutoff, "oos_date": oos_idx, "y_true": y_true, "y_pred": y_pred, **quantiles})
        )

    return pd.concat(out, ignore_index=True).sort_values(["cutoff", "oos_date"]).set_index(["cutoff", "oos_date"])


def prepare_data(
    df: pd.DataFrame,
    y: str,
    X: list[str] | None = None,
    horizon: int = 1,
    oos_start: str = "2020-01-31",
    freq: str | None = None,
) -> TestData:
    if freq is None:
        freq = infer_freq(df)

    ds = PandasDataset(dataframes=df, target=y, freq=freq, past_feat_dynamic_real=X)

    # generate <horizon> rolling fc for each <window=oos_date>, and move by <distance> forward
    n_oos = int((df.index >= oos_start).sum())
    _, test_tmpl = split(ds, offset=-n_oos)
    return test_tmpl.generate_instances(prediction_length=horizon, windows=n_oos - horizon + 1, distance=1)


class Moirai(Model, name="moirai"):
    @staticmethod
    def get_backbone():
        return MoiraiModule.from_pretrained(MODEL_ID)

    def get_model(self, ctx_len: int, horizon: int, n_covariates: int) -> MoiraiForecast:
        return MoiraiForecast(
            module=self.get_backbone(),
            prediction_length=horizon,
            context_length=ctx_len,
            patch_size="auto",
            num_samples=1_000,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=n_covariates,
        )

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
        test_data = prepare_data(df, y, X, horizon, oos_start, freq)
        mdl = self.get_model(ctx_len, horizon, n_covariates=len(X) if X else 0)

        predictor = mdl.create_predictor(batch_size=256, device="auto")
        forecasts = list(predictor.predict(test_data.input))
        return make_fc_df(forecasts, df[y], freq)
