import os
from typing import List, Dict

from ..utils.mol_utils import get_molecule_scaffold
import torch
from torch.utils.data import Dataset
from tqdm import tqdm



class ScaffoldDataset(Dataset):

    def __init__(self,
                 data_path: str,
                 tokenizer,
                 return_scaffold: bool=True,
                 max_len: int=0) -> None:

        self.max_len = max_len
        self.data_path = data_path 

        self._molecules = self.load_molecules()
        
        self.scaffolds = list(set([get_molecule_scaffold(mol) for mol in tqdm(self._molecules, desc='generating scaffolds')])) 
        
        self.tokenizer = tokenizer

        self.return_scaffold = return_scaffold

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
        encodings = {}
        
        smiles = self._molecules[idx]
        encoded_smiles = '[BOS]' + smiles + '[EOS]'
        smiles_encodings = self.tokenizer(encoded_smiles, padding=True, max_length=self.max_len)
        
        encodings['dec_inp'] = smiles_encodings['input_ids']
        encodings['labels'] = smiles_encodings['input_ids']
        encodings['dec_padding_mask'] = smiles_encodings['padding_mask']
        

        if self.return_scaffold:
            scaffold = get_molecule_scaffold(smiles)
            scaffold = '[BOS]' + scaffold + '[EOS]'
            scaffold_encodings = self.tokenizer(scaffold, padding=True, max_length=self.max_len)
            
            encodings['enc_inp'] = scaffold_encodings['input_ids']
            encodings['enc_padding_mask'] = scaffold_encodings['padding_mask']

        else:
            scaffold_encodings = self.tokenizer('', padding=True, max_length=self.max_len)
            
            encodings['enc_inp'] = scaffold_encodings['input_ids']
            encodings['enc_padding_mask'] = scaffold_encodings['padding_mask']
        
        encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        return encodings

def main():

    dataset = ScaffoldDataset('./data/gdb/gdb13/full/1.smi',
                                 './data/tokenizers/gdb13FullCharTokenizer.json')

    smiles = 'CCO'
    
    encoding = dataset.tokenizer.encode(smiles)
    print(encoding)
    rec_smiles = dataset.tokenizer.decode(encoding)
    print(rec_smiles)
    print(rec_smiles == smiles)

if __name__ == "__main__":
    main()
