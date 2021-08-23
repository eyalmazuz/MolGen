import json
import os
from typing import List, Dict

from rdkit import Chem
from rdkit.Chem.rdmolfiles import MultithreadedSmilesMolSupplier, SmilesMolSupplier
from rdkit import RDLogger

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

class CharSmilesDataset(Dataset):

    def __init__(self, data_path: str, tokenizer_path: str=None) -> None:

        self.molecules = MultithreadedSmilesMolSupplier(data_path)
        if tokenizer_path and os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'r') as f:
                self.id2token = json.load(f)
                self.id2token = {int(k): v for k, v in self.id2token.items()}
        else:
            self.id2token = self.build_tokenizer(tokenizer_path)

        self.token2id = {v: k for k, v in self.id2token.items()}
        self.max_len = self.get_max_smiles_len()

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx):

        mol = self.molecules[idx]
        mol_smiles = Chem.MolToSmiles(mol)
        tokens = self.tokenizer.encode(mol_smiles)

        return tokens
    
    def get_max_smiles_len(self, ) -> int:
        max_len = 0
        for mol in tqdm(self.molecules):
            if mol:
                smiles = Chem.MolToSmiles(mol)
                if len(smiles) > max_len:
                    max_len = len(smiles)
        return max_len

    def build_tokenizer(self, tokenizer_path: str) -> Dict[int, str]:
        print('Building tokenzier')

        tokens = set()
        for mol in tqdm(self.molecules):
            if mol:
                smiles = Chem.MolToSmiles(mol)
                tokens |= set(smiles)

        id2token = {}
        for i, token in enumerate(tokens):
            id2token[i] = token

        len_tokens = len(id2token)

        id2token[len_tokens] = '[SEP]'
        id2token[len_tokens + 1] = '[UNK]'
        id2token[len_tokens + 2] = '[PAD]'
        id2token[len_tokens + 3] = '[BOS]'
        id2token[len_tokens + 4] = '[EOS]'
        
        print('Saving tokenizer')
        if tokenizer_path:
            with open(tokenizer_path, 'w') as f:
                json.dump(id2token, f)

        return id2token

    def encode(self, smiles: str) -> List[int]:
        encodings = []
        for char in smiles:
            encodings.append(self.token2id[char])
        
        if len(encodings) < self.max_len:
            encodings += [self.token2id['[SEP]']] * (self.max_len - len(encodings))
        return [self.token2id['[BOS]']] + encodings + [self.token2id['[EOS]']]
    
    def decode(self, encodings: List[int]) -> str:
        chars = []
        for id_ in encodings:
            chars.append(self.id2token[id_])

        return ''.join(chars)

class BPESmilesDataset(Dataset):

    def __init__(self, encodings) -> None:
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.encodings['input_ids'])
        return item


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
