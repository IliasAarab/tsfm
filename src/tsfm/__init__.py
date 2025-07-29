import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from tsfm.data import generator
from tsfm.models.ols import ols_estimator
from tsfm.vis import plot_preds

plt.style.use(["science", "no-latex", "notebook"])
