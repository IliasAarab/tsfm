import matplotlib.pyplot as plt


def plot_preds(df, y_pred, title="OOS predictions"):
    """
    Plot the full true series and highlight OOS predictions vs. true in the OOS period.

    Args:
        df (pd.DataFrame): DataFrame with the full true series ('y').
        y_pred (array-like): Out-of-sample predictions, assumed to align with the last N rows.
        title (str): Plot title.
    """
    n_oos = len(y_pred)
    oos_start = df.index[-n_oos]

    oos_index = df.index[-n_oos:]
    oos_true = df["y"].iloc[-n_oos:]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["y"], color="gray", label="True (All)", linewidth=1)
    ax.axvline(oos_start, color="red", linestyle="--", label="OOS Start")

    ax.plot(oos_index, oos_true, label="True (OOS)", color="black", marker="o")
    ax.plot(oos_index, y_pred, label="Predicted (OOS)", color="tab:blue", marker="x")

    ax.set_xlabel("Time Index")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return ax
