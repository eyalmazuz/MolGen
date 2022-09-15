from random import sample
from typing import List, Dict

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils.mol_utils import get_molecule_scaffold



class SmilesDataset(Dataset):

    def __init__(self,
                 data_path: str,
                 tokenizer,
                 use_scaffold=False,
                 max_len: int=0) -> None:

        self.max_len = max_len
        self.data_path = data_path 
        self.use_scaffold = use_scaffold

        self._molecules = self.load_molecules()

        if self.use_scaffold:
            self.scaffolds = list(set([get_molecule_scaffold(mol) for mol in tqdm(self._molecules[:100000], desc='generating scaffolds')]))

        self.tokenizer = tokenizer

    @property
    def molecules(self) -> List[str]:
        return self._molecules
        
    def load_molecules(self,) -> List[str]:
        molecules = []
        with open(self.data_path, 'r') as f:
            molecules = f.readlines()
            molecules = [smiles.strip() for smiles in molecules]
        return molecules

    def __len__(self) -> int:
        return len(self._molecules)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        smiles = self._molecules[idx]
        
        if self.use_scaffold:
            scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles)
            smiles = '[BOS]' + scaffold + '[SEP]' + smiles + '[EOS]' 
        else:
            smiles = '[BOS]' + smiles + '[EOS]'
        encodings = self.tokenizer(smiles, padding=True, max_length=self.max_len)
        encodings['labels'] = encodings['input_ids']

        encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        return encodings

def main():

    dataset = SmilesDataset('./data/gdb/gdb13/full/1.smi',
                                 './data/tokenizers/gdb13FullCharTokenizer.json')

    smiles = 'CCO'
    
    encoding = dataset.tokenizer.encode(smiles)
    print(encoding)
    rec_smiles = dataset.tokenizer.decode(encoding)
    print(rec_smiles)
    print(rec_smiles == smiles)

if __name__ == "__main__":
    main()
