from enum import Enum, auto


class ModelType(Enum):
    GPT = auto()
    BERT = auto()

    @classmethod
    def from_str(label: str) -> "ModelType":
        if label.lower() == "gpt":
            return ModelType.GPT
        elif label.lower() == "bert":
            return modelType.BERT

