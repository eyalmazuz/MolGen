from enum import Enum, auto
import operator
from typing import Callable


class AggType(Enum):
    ADD = auto()
    MUL = auto()

    @staticmethod
    def from_str(option: str) -> "AggType":
        if option.upper() == "ADD":
            return AggType.ADD
        elif option.upper() == "MUL":
            return AggType.MUL
        else:
            raise NotImplementedError


def agg_to_op(agg_type: str) -> Callable[[float, float], float]:
    agg = AggType.from_str(agg_type)

    if agg == AggType.ADD:
        return operator.add
    elif agg == AggType.MUL:
        return operator.mul
    else:
        raise ValueError

