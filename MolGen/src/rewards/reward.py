from enum import Enumfrom abc import ABC, abstractmethod

from typing import Callable, Dict, List, Optional, Union

class AbstractReward(ABC):
    def __init__(self,
                 name: Optional[str]=None,
                 scale: Optional[Callable[[float], float]]=None,
                 eval_: bool=False) -> None:
        self.name = name
        self.scale = scale
        self._eval = eval_

    @abstractmethod
    def __call__(self, smiles: Union[str, List[str]]) -> Union[float, List[float]]:
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
        return f"{self.__class__.__name__}_{self.name=}_{str(self.scale)=}_{self._eval=}"
