import matplotlib.pyplot as plt
import scienceplots

from tsfm.data import generator
from tsfm.models.armodel import ARModel
from tsfm.models.base import Model
from tsfm.models.moirai import Moirai
from tsfm.vis import plot_preds

plt.style.use(["science", "no-latex", "notebook"])
