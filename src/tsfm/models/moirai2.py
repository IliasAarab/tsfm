import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import TestData, split
from pandas.tseries.frequencies import to_offset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from tsfm.models.base import Model

MODEL_ID = "Salesforce/moirai-2.0-R-small"


def make_fc_df(forecasts, y_true_series: pd.Series) -> pd.DataFrame:
    out = []
    for fc in forecasts:
        off = to_offset("M")
        start = fc.start_date.to_timestamp(how="end").date()
        horizon = len(fc.quantile(0.5))
        oos_idx = pd.date_range(start=start, periods=horizon, freq=off)
        cutoff = start - off

        y_pred = fc.quantile(0.5)  # median as point forecast
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
) -> TestData:
    ds = PandasDataset(dataframes=df, target=y, freq="M", past_feat_dynamic_real=X)
    n_oos = int(sum(df.index >= oos_start))
    _, test_tmpl = split(ds, offset=-n_oos)
    return test_tmpl.generate_instances(prediction_length=horizon, windows=n_oos - horizon + 1, distance=1)


class Moirai2(Model, name="moirai2"):
    @staticmethod
    def get_backbone():
        return Moirai2Module.from_pretrained(MODEL_ID)

    def get_model(self, ctx_len: int, horizon: int, n_covariates: int) -> Moirai2Forecast:
        return Moirai2Forecast(
            module=self.get_backbone(),
            prediction_length=horizon,
            context_length=ctx_len,
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
        test_data = prepare_data(df, y, X, horizon, oos_start)
        mdl = self.get_model(ctx_len, horizon, n_covariates=len(X) if X else 0)
        predictor = mdl.create_predictor(batch_size=256)
        forecasts = list(predictor.predict(test_data.input))
        return make_fc_df(forecasts, df[y])
