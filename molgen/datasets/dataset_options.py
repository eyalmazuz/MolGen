from enum import Enum, auto


class DatasetType(Enum):
    SMILES = auto()
    DT_SMILES = auto()

    @staticmethod
    def from_str(label: str) -> "DatasetType":
        if label.upper() == "SMILES":
            type_ = DatasetType.SMILES
        elif label.upper() == "DT_SMILES":
            type_ = DatasetType.DT_SMILES
        else:
            raise ValueError(f"Invalid type: {label}")

        return type_
