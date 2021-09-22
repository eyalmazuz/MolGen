import json
import os
from typing import Dict, List

from tqdm import tqdm


class CharTokenizer():
    
    def __init__(self, tokenizer_path: str=None, data_path=None):

        if tokenizer_path and os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'r') as f:
                self.id2token = json.load(f)
                self.id2token = {int(k): v for k, v in self.id2token.items()}
        else:
            self.id2token = self.build_tokenizer(data_path, tokenizer_path)

        self.token2id = {v: k for k, v in self.id2token.items()}
       
    
    @property
    def vocab_size(self):
        return len(self.id2token)

    @property
    def bos_token(self):
        return '[BOS]'

    @property
    def bos_token_id(self):
        return self.token2id['[BOS]']

    @property
    def eos_token(self):
        return '[EOS]'

    @property
    def eos_token_id(self):
        return self.token2id['[EOS]']

    @property
    def pad_token(self):
        return '[PAD]'

    @property
    def pad_token_id(self):
        return self.token2id['[PAD]']


    def build_tokenizer(self, tokenizer_path: str, data_path:str) -> Dict[int, str]:

        with open(data_path, 'r') as f:
            self.molecules = f.readlines()
            self.molecules = [smiles.strip() for smiles in self.molecules]

        print('Building tokenzier')

        tokens = set()
        for smiles in tqdm(self.molecules):
            if smiles:
                tokens |= set(smiles)
        print(tokens)

        id2token = {}
        for i, token in enumerate(tokens):
            id2token[i] = token

        len_tokens = len(id2token)

        id2token[len_tokens + 0] = '[PAD]'
        id2token[len_tokens + 1] = '[BOS]'
        id2token[len_tokens + 2] = '[EOS]'
        
        print('Saving tokenizer')
        if tokenizer_path:
            with open(tokenizer_path, 'w') as f:
                json.dump(id2token, f)

        return id2token


    def tokenize(self, smiles, padding=None, max_length=None):
        encodings = []
        bos, eos = None, None
        if smiles.startswith('[BOS]'):
            bos = self.bos_token_id
            smiles = smiles[5:]
        
        if smiles.endswith('[EOS]'):
            eos = self.eos_token_id
            smiles = smiles[:-5]

        for char in smiles:
            encodings.append(self.token2id[char])
        
        if bos:
            encodings = [bos] + encodings

        if eos:
            encodings = encodings + [eos]
        attention_mask = [1] * len(encodings)
        
        if padding and max_length and len(encodings) < max_length:
            pad_len = (max_length - len(encodings))
            encodings += [self.token2id['[PAD]']] * pad_len
            attention_mask += [0] * pad_len 

        return {"input_ids": encodings,
                "attention_mask": attention_mask}

 
    def __call__(self, smiles, padding=False, max_length=None):
        return self.tokenize(smiles, padding, max_length)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        encodings = []
        for char in tokens:
            encodings.append(self.token2id[char])
        
        return [encodings]

    def convert_ids_to_tokens(self, encodings: List[int]) -> List[str]:
        tokens = []
        for id_ in encodings:
            tokens.append(self.id2token[id_])

        return tokens


    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)


    def decode(self, tokens: List[int]) -> str:
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(tokens))

def main():
    tokenizer = CharTokenizer('../../../data/tokenizers/gdb13CharTokenizer.json')
    
    print(tokenizer.vocab_size)
    smiles = '(CC1C(C)C(C(CC#N)C=C)C1)C'
    encodings = tokenizer(smiles, max_length=40)
    print(encodings)
    decoded = tokenizer.decode(encodings['input_ids'])
    print(decoded)

if __name__ == "__main__":
    main()
