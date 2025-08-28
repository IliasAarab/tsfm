# %% Setup
import matplotlib.pyplot as plt
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

from tsfm import generator, plot_preds

# %% Hyperparams
N_OOS = 10
MODEL_ID = "Salesforce/moirai-1.1-R-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_LEN = 200

# %% Data
df = generator()
df.index = pd.date_range("2000-01", periods=len(df), freq="M")
print("Snippet of dataset:")
print(df.head())

ds = PandasDataset(
    dataframes=df,
    target="y",
    freq="M",
    past_feat_dynamic_real=["x"],
    future_length=N_OOS,
)

train, test_tmpl = split(ds, offset=-N_OOS)
test_data = test_tmpl.generate_instances(
    prediction_length=N_OOS,
    windows=1,
    distance=N_OOS,
)


# %% Preds
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base"),
    prediction_length=N_OOS,
    context_length=CONTEXT_LEN,
    patch_size="auto",
    num_samples=200,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

predictor = model.create_predictor(batch_size=32)
forecasts = list(predictor.predict(test_data.input))

fc = forecasts[0]
p50 = fc.quantile(0.5)
p10 = fc.quantile(0.1)
p90 = fc.quantile(0.9)
mean_pred = fc.mean

# %% Plot
ax = plot_preds(df, p50, title="OOS predictions: Moirai")
ax.fill_between(df.index[-N_OOS:], p10, p90, alpha=0.3, color="C0", label="80% PI")
ax.plot(df.index[-N_OOS:], mean_pred, color="C1", linestyle="--", label="Mean pred")
ax.legend()
plt.savefig("figs/3.moirai_preds.png")
