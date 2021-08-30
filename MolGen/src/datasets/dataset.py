import json
import os
from typing import List, Dict

from rdkit import Chem
from rdkit.Chem.rdmolfiles import SmilesMolSupplier#, MultithreadedSmilesMolSupplier 
from rdkit import RDLogger

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

class SmilesDataset(Dataset):

    def __init__(self, data_path: str, tokenizer) -> None:

        with open(data_path, 'r') as f:
            self.molecules = f.readlines()
            self.molecules = [smiles.strip() for smiles in self.molecules]
        
        self.tokenizer = tokenizer
        
        self.max_len = self.get_max_smiles_len()

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx):

        smiles = self.molecules[idx]
        smiles = '[BOS]' + smiles + '[EOS]'
        encodings = self.tokenizer(smiles, padding=True, max_length=self.max_len)
        encodings['labels'] = encodings['input_ids']
        #encodings['input_ids'] = encodings['input_ids']
        encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        return encodings
    
    def get_max_smiles_len(self, ) -> int:
        max_len = 0
        for smiles in tqdm(self.molecules):
            if len(smiles) > max_len:
                max_len = len(smiles)
        return max_len + 2

def main():

    dataset = CharSmilesDataset('../../../data/gdb/gdb13/gdb13.rand1M.smi',
                                 '../../../data/tokenizers/gdb13CharTokenizer.json')

    smiles = 'Cc1ncc(C[n+]2csc(CCO)c2C)c(N)n1'
    
    encoding = dataset.encode(smiles)
    print(encoding)
    rec_smiles = dataset.decode(encoding)
    print(rec_smiles)
    print(rec_smiles == smiles)

if __name__ == "__main__":
    main()
