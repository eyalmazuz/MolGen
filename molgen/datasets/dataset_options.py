from enum import Enum, auto


class DatasetType(Enum):
    SMILES = auto()
    DT_SMILES = auto()
    SELFIES = auto()
    DT_SELFIES = auto()

    @staticmethod
    def from_str(label: str) -> "DatasetType":
        if label.upper() == "SMILES":
            type_ = DatasetType.SMILES
        elif label.upper() == "DT_SMILES":
            type_ = DatasetType.DT_SMILES
        elif label.upper() == "SELFIES":
            type_ = DatasetType.SELFIES
        elif label.upper() == "DT_SELFIES":
            type_ = DatasetType.DT_SELFIES
        else:
            raise ValueError(f"Invalid type: {label}")

        return type_
