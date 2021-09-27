import json
import linecache
import os
from typing import List, Dict

from rdkit import Chem
from rdkit import RDLogger

import torch
from torch.utils import data
from torch.utils.data import Dataset
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

class SmilesDataset(Dataset):

    def __init__(self, data_path: str, tokenizer, max_len: int=0) -> None:

        self.max_len = max_len
        self.len = 0
        with open(data_path, 'r') as f:
            for line in f:
                self.len += 1

        self.data_path = data_path        
        # with open(data_path, 'r') as f:
        #     self.molecules = f.readlines()
        #     self.molecules = [smiles.strip() for smiles in self.molecules]
        
        # self.test_molecules = self.molecules[int(len(self.molecules) * 0.8):]
        # self.molecuels = self.molecules[:int(len(self.molecules) * 0.8)]
        self.tokenizer = tokenizer
        

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx):
        smiles = linecache.getline(self.data_path, idx + 1).strip()
        smiles = '[BOS]' + smiles + '[EOS]'
        encodings = self.tokenizer(smiles, padding=True, max_length=self.max_len)
        encodings['labels'] = encodings['input_ids']
        #encodings['input_ids'] = encodings['input_ids']
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
