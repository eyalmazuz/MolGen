from random import sample
from typing import List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset



class BS1Dataset(Dataset):

    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer,
                 max_len: int=0) -> None:

        self.max_len = max_len

        self.data = data

        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        smiles = self.data.loc[idx]['Smiles']
        label = self.data.loc[idx]['pChEMBL Value'] 

        smiles = '[BOS]' + smiles + '[EOS]'
        encodings = self.tokenizer(smiles, padding=True, max_length=self.max_len)

        encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        return encodings, label
