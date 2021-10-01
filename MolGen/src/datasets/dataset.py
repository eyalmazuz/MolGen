import json
import linecache
import os
from typing import List, Dict, Union

from rdkit import Chem
from rdkit import RDLogger

import torch
from torch._C import _supported_qengines
from torch.utils import data
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

class SmilesDataset(Dataset):

    def __init__(self,
                 data_path: str,
                 tokenizer,
                 max_len: int=0,
                 to_load: bool=False) -> None:

        self.max_len = max_len
        self.len = 0
        with open(data_path, 'r') as f:
            for _ in f:
                self.len += 1
        self.to_load = to_load
        self.data_path = data_path 
        
        if self.to_load:
            self._molecules = self.load_molecules()       

        self.tokenizer = tokenizer

    @property
    def molecules(self) -> List[str]:
        if self.to_load:
            molecules = self._molecules
        else:
            molecules = self.load_molecules()
        
        return molecules
        
    def load_molecules(self,) -> List[str]:
        molecules = []
        with open(self.data_path, 'r') as f:
            molecules = f.readlines()
            molecules = [smiles.strip() for smiles in molecules]
        return molecules

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.to_load:
            smiles = self._molecules[idx]
        else:
            smiles = linecache.getline(self.data_path, idx + 1).strip()
        smiles = '[BOS]' + smiles + '[EOS]'
        encodings = self.tokenizer(smiles, padding=True, max_length=self.max_len)
        encodings['labels'] = encodings['input_ids']
        #encodings['input_ids'] = encodings['input_ids']
        encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        return encodings

def get_dataset(data_path: str,
                tokenizer, max_len:int,
                to_load: bool=False) -> Union[Dataset, ConcatDataset]:
    
    if os.path.isdir(data_path):
        datasets = []    
        for path in os.listdir(data_path):
            full_path = os.path.join(data_path, path)
            dataset = SmilesDataset(full_path,
                                    tokenizer,
                                    max_len=max_len,
                                    to_load=to_load)
            datasets.append(dataset)

        dataset = ConcatDataset(datasets)
    
    else:
        dataset = SmilesDataset(data_path,
                                tokenizer,
                                max_len=max_len,
                                to_load=to_load)
    return dataset

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
