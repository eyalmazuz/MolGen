from rdkit import Chem
from rdkit.Chem.rdmolfiles import MultithreadedSmilesMolSupplier, SmilesMolSupplier

from torch.utils.data import Dataset, DataLoader

class SmilesDataset(Dataset):

    def __init__(self, data_path: str, tokenizer) -> None:

        self.molecules = SmilesMolSupplier(data_path)
        self.tokenzier = tokenizer

    def __len__(self) -> int:
        return len(self.molecules)

    
    def __getitem__(self, idx):

        mol = self.molecules[idx]
        mol_smiles = Chem.MolToSmiles(mol)
        tokens, attention_mask = self.tokenizer.encode(mol_smiles)

        return {"input_ids": tokens, "attention_masl": attention_mask}


