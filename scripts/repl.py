# %%
import pandas as pd

from tsfm import Model, generator

df = generator()

mdl = Model.build(name="armodel")
yhs = mdl.pred(df, y="y", X=None, horizon=1, oos_start="2005-01-31")

# %%

cols_to_drop = "Negotiated wages including"
df = pd.read_csv("data.csv").dropna().rename(columns={"Unnamed: 0": "dt"})
df["dt"] = pd.to_datetime(df["dt"], format="%Y%b").add(pd.offsets.MonthEnd(0))
df.index = df.pop("dt")
df = df.drop(cols_to_drop, axis=1).astype(float).sort_index()

y = "Negotiated wages excluding"
X = list(df.columns[1:])
yhs = mdl.pred(df, y=y, X=None, horizon=2, oos_start="2005-01-31")

yhs.head(20).to_clipboard()

# %%
import numpy as np
import pandas as pd

# assume df is your DataFrame with MultiIndex [cutoff, oos_date]
cut = yhs.index.get_level_values(0)
oos = yhs.index.get_level_values(1)

# convert to monthly period ordinals to compute horizon
cut_ord = cut.to_period("M").astype("int64")
oos_ord = oos.to_period("M").astype("int64")
h = (oos_ord - cut_ord) + 1

# compute squared errors
se = (yhs["y_pred"] - yhs["y_true"]) ** 2

# group by horizon and compute RMSFE
rmsfe_per_h = se.groupby(h).mean().pipe(np.sqrt)
rmsfe_per_h.index.name = "horizon"

# average across horizons
avg_rmsfe = float(rmsfe_per_h.mean())

print(rmsfe_per_h)
print("avg_rmsfe:", avg_rmsfe)


# %%
import matplotlib.pyplot as plt
import pandas as pd


def plot_fc_by_horizon(
    df: pd.DataFrame,
    horizon: int = 1,
) -> plt.Axes:
    """
    Plot y_true vs y_pred for a given forecast horizon.
    Assumes a MultiIndex with levels [cutoff, oos_date] (order configurable).

    horizon=1 -> first oos_date per cutoff, horizon=2 -> second, etc.
    Returns the matplotlib Axes.
    """

    # One row per cutoff at the requested horizon
    sel = df.groupby(level=0, sort=False).nth(horizon - 1, dropna="all")

    # X-axis = oos_date level if still present, else the current index
    x = sel.index.get_level_values(1)

    fig, ax = plt.subplots()

    ax.plot(x, sel["y_true"], label="y_true", linewidth=2)
    ax.plot(x, sel["y_pred"], label="y_pred", linewidth=1.8)
    ax.set_title(f"y_true vs y_pred @ horizon={horizon}")
    ax.set_xlabel("oos_date")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


plot_fc_by_horizon(yhs)
