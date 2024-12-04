from abc import ABC, abstractmethod
from functools import partial
from typing import Any

from molgen.rewards import reward_scales

RewardScale = str | dict[str, Any] | None

class AbstractReward(ABC):
    def __init__(self,
                 name: str | None=None,
                 scale: RewardScale=None,
                 eval_: bool=False) -> None:
        self.name = name
        self._eval = eval_
        self.scale: partial[Any] | None

        if scale is not None:
            if isinstance(scale, dict):
                func_name = scale.pop("name")
                kwargs = scale
            elif isinstance(scale, str):
                func_name = scale
                kwargs = {}
            else:
                raise ValueError("Invalid scale config")

            if not hasattr(reward_scales, func_name):
                raise ValueError(f"{func_name} is not defined in reward scales")

            func = getattr(reward_scales, func_name)
            self.scale = partial(func, **kwargs)
        else:
            self.scale = None


    @abstractmethod
    def __call__(self, smiles: str | list[str]) -> float | list[float]:
        pass

    @property
    def eval(self) -> bool:
        return self._eval

    @eval.setter
    def eval(self, val: bool) -> None:
        if isinstance(val, bool):
            self._eval = val
        else:
            raise ValueError("Can only set eval to boolean")

    def __str__(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return self.__class__.__name__

    def __repr__(self) -> str:
        scale_str = str(self.scale) if self.scale is not None else ""
        return f"{self.__class__.__name__}_self.scale={scale_str}_{self._eval=}"
