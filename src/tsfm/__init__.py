import matplotlib.pyplot as plt
import scienceplots

from tsfm.data import generator
from tsfm.models.armodel import ARModel
from tsfm.models.base import Model
from tsfm.models.moirai import Moirai

plt.style.use(["science", "no-latex", "notebook"])

__all__ = ["ARModel", "Model", "Moirai", "generator"]
