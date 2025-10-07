from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd


class Model(ABC):
    registry: ClassVar[dict[str, type["Model"]]] = {}

    def __init_subclass__(cls, name: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.arch = property(lambda self: name)
        Model.registry[name] = cls

    @classmethod
    def build(cls, *, name: str, **kwargs):
        return cls.registry[name](**kwargs)

    @abstractmethod
    def pred(
        self,
        df: pd.DataFrame,
        y: str,
        X: list[str] | None = None,
        ctx_len: int = 1,
        horizon: int = 1,
        oos_start: str = "2020-01-31",
    ) -> pd.DataFrame: ...
