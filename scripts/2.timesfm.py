# %% Setup
import matplotlib.pyplot as plt
import torch
from transformers import TimesFmModelForPrediction

from tsfm import generator, plot_preds

# %% Hyperparams
N_OOS = 10
MODEL_ID = "google/timesfm-2.0-500m-pytorch"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_LEN = 32

# %% Data
df = generator()
print("Snippet of dataset:")
print(df.head())

# %% Preds
past = torch.tensor(
    df["y"].values[:-N_OOS][-CONTEXT_LEN:], dtype=torch.bfloat16, device=DEVICE
).unsqueeze(0)  # (1, T)
freq = torch.tensor([0], dtype=torch.long, device=DEVICE)

model = TimesFmModelForPrediction.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=DEVICE
).eval()

with torch.no_grad():
    out = model(
        past_values=past,
        freq=freq,
        forecast_context_len=CONTEXT_LEN,
        truncate_negative=False,
        return_dict=True,
    )

# ── extract statistics ────────────────────────────────────────────────
full = out.full_predictions[0, :, :N_OOS].float().cpu().numpy()  # (Q+1, N_OOS)

q_list = model.config.quantiles
idx_lo = q_list.index(0.1)
idx_med = q_list.index(0.5)
idx_hi = q_list.index(0.9)

mean_pred = full[0]
median_pred = full[1 + idx_med]
lower = full[1 + idx_lo]
upper = full[1 + idx_hi]

# %% Plot results
ax = plot_preds(df, median_pred, title="OOS predictions: TimesFM")
ax.fill_between(df.index[-N_OOS:], lower, upper, alpha=0.3, color="C0", label="80% PI")
ax.plot(df.index[-N_OOS:], mean_pred, color="C1", linestyle="--", label="Mean pred")
ax.legend()
plt.savefig("figs/2.timesfm_preds.png")
