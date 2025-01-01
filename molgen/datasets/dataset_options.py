from enum import StrEnum, auto


class DatasetType(StrEnum):
    SMILES = auto()
    DT_SMILES = auto()
    SELFIES = auto()
    DT_SELFIES = auto()
