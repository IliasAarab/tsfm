# %% Setup
from tsfm import generator, ols_estimator, plot_preds
import matplotlib.pyplot as plt

# %% Hyperparams
N_OOS = 10  

# %% Data
df = generator()
print("Snippet of dataset:")
print(df.head())
df.plot(title=r"Overview dynamic system: $y=f(x, y)$")
plt.savefig("figs/1.1.overview_dynamic_system.png")

# %% Preds
yhs = ols_estimator(df)
plot_preds(df, yhs)
plt.savefig("figs/1.2.ar_preds.png")



