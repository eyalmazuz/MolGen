from enum import Enum, auto


class ModelType(Enum):
    GPT = auto()
    BERT = auto()
    TRANSFORMER = auto()
    RNN = auto()

    @staticmethod
    def from_str(label: str) -> "ModelType":
        match label.upper():
            case "GPT":
                type_ = ModelType.GPT
            case "BERT":
                type_ = ModelType.BERT
            case "TRANSFORMER":
                type_ = ModelType.TRANSFORMER
            case "RNN":
                type_ = ModelType.RNN
            case _:
                raise ValueError(f"Invalid type: {label}")

        return type_
