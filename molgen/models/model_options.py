from enum import Enum, auto


class ModelType(Enum):
    GPT = auto()
    BERT = auto()
    TRANSFORMER = auto()
    RNN = auto()

    @staticmethod
    def from_str(label: str) -> "ModelType":
        if label.upper() == "GPT":
            type_ = ModelType.GPT
        elif label.upper() == "BERT":
            type_ = ModelType.BERT
        elif label.upper() == "TRANSFORMER":
            type_ = ModelType.TRANSFORMER
        elif label.upper() == "RNN":
            type_ = ModelType.RNN

        return type_

