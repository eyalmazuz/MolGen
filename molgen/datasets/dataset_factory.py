from torch.utils.data import Dataset

from molgen.datasets.dataset_options import DatasetType
from molgen.datasets.smiles_dataset import PreTrainGPTSmilesDataset, PreTrainDecisionGPTSmilesDataset

from molgen.models.model_options import ModelType


def get_dataset(dataset_type: DatasetType, model_type: ModelType, **kwargs) -> Dataset:
    match model_type:
        case ModelType.GPT:
            dataset = get_gpt_dataset(dataset_type, **kwargs)
        case _:
            raise ValueError(f"Invalid model type {model_type}")

    return dataset


def get_gpt_dataset(dataset_type: DatasetType, **kwargs) -> Dataset:
    match dataset_type:
        case DatasetType.SMILES:
            dataset = PreTrainGPTSmilesDataset(**kwargs)
        case DatasetType.DT_SMILES:
            dataset = PreTrainDecisionGPTSmilesDataset(**kwargs)

    return dataset


if __name__ == '__main__':
    from molgen.tokenizers.tokenizer_factory import get_tokenizer
    import os
    tokenizer = get_tokenizer(os.path.join(os.path.dirname(os.getcwd()), "data", "bpeTokenizer"))
    dataset_path = "/mnt/c/Users/Or/PycharmProjects/Thesis/MolGen/data/datasets/250k_rndm_zinc_drugs_clean_3.csv"
    get_gpt_dataset(DatasetType.SMILES, dataset_path=dataset_path, tokenizer=tokenizer)
    pass
