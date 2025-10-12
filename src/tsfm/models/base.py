from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd

from tsfm.outputs import ForecastOutput


class Model(ABC):
    registry: ClassVar[dict[str, type["Model"]]] = {}

    def __init_subclass__(cls, name: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.name = property(lambda self: name)  # noqa: ARG005
        Model.registry[name] = cls

    @classmethod
    def build(cls, *, name: str, **kwargs):
        print(f"Building {name}...", end=" ")  # noqa: T201
        mdl = cls.registry[name](**kwargs)
        mdl.get_backbone()
        print(f"{name} has been build!")  # noqa: T201
        return mdl

    @abstractmethod
    def get_backbone(self, *args, **kwargs) -> Any: ...

    @abstractmethod
    def _pred(
        self,
        df: pd.DataFrame,
        y: str,
        X: list[str] | None = None,
        ctx_len: int = 1,
        horizon: int = 1,
        oos_start: str = "2020-01-31",
    ) -> pd.DataFrame: ...

    def pred(
        self,
        df: pd.DataFrame,
        y: str,
        X: list[str] | None = None,
        ctx_len: int = 1,
        horizon: int = 1,
        oos_start: str = "2020-01-31",
    ) -> ForecastOutput:
        yhs = self._pred(df, y, X, ctx_len, horizon, oos_start)
        meta = {"model": self.name, "oos_start": oos_start}
        return ForecastOutput(yhs, meta=meta)
