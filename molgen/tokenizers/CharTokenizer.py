import json
import os
from typing import Dict, List, Union

from tqdm import tqdm


class CharTokenizer():
    
    def __init__(self, tokenizer_path: str='./tokenizers/', data_path: str='./data/', **kwargs) -> None:

        print(f'{tokenizer_path=}', f'{os.path.exists(tokenizer_path)=}')
        if tokenizer_path and os.path.exists(tokenizer_path):
            print('Loading Existing tokenizer')
            with open(tokenizer_path, 'r') as f:
                self.id2token = json.load(f)
                self.id2token = {int(k): v for k, v in self.id2token.items()}
        else:
            print('Building tokenizer')
            self.id2token = {}
            if os.path.isdir(data_path):
                for path in os.listdir(data_path):
                    full_path = os.path.join(data_path, path)
                    tokenized_file = self.build_tokenizer(full_path)
                    self.id2token = {**self.id2token, **tokenized_file}
                    print(self.id2token)
            else:
                self.id2token = self.build_tokenizer(data_path)
                
            len_tokens = len(self.id2token)

            self.id2token[len_tokens + 0] = '[PAD]'
            self.id2token[len_tokens + 1] = '[BOS]'
            self.id2token[len_tokens + 2] = '[EOS]'
            self.id2token[len_tokens + 3] = '[SEP]'
            self.id2token[len_tokens + 4] = '[UNK]'
            self.id2token[len_tokens + 5] = '[CLS]'
            print(self.id2token)
            
            print('Saving tokenizer')
            if tokenizer_path:
                with open(tokenizer_path, 'w') as f:
                    json.dump(self.id2token, f)

        self.token2id: Dict[str, int] = {v: k for k, v in self.id2token.items()}
       
        print(self.id2token)
        print(self.token2id)
        
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

    @property
    def sep_token(self):
        return '[SEP]'

    @property
    def sep_token_id(self):
        return self.token2id['[SEP]']

    @property
    def unk_token(self):
        return '[UNK]'

    @property
    def unk_token_id(self):
        return self.token2id['[UNK]']

    @property
    def cls_token(self):
        return '[CLS]'

    @property
    def cls_token_id(self):
        return self.token2id['[CLS]']


    def build_tokenizer(self, data_path:str) -> Dict[int, str]:

        with open(data_path, 'r') as f:
            molecules = f.readlines()
            molecules = [smiles.strip() for smiles in molecules]

        print('Building tokenzier')

        tokens = set()

        for mol in molecules:
            tokens |= set(mol)

        id2token = {}
        for i, token in enumerate(tokens):
            id2token[i] = token

        return id2token


    def tokenize(self, smiles, padding: bool=False, max_length: int=-1):
        encodings = []
        bos, eos, sep, cls, sca = [], [], [], [], []
         
        if smiles.startswith('[CLS]'):
            smiles = smiles[5:]
            cls.append('[CLS]')   
        
        if smiles.startswith('[BOS]'):
            smiles = smiles[5:]
            bos.append('[BOS]')   
                 
        if smiles.endswith('[EOS]'):
            eos.append('[EOS]') 
            smiles = smiles[:-5]

        if '[SEP]' in smiles:
            idx = smiles.find('[SEP]')
            sca += smiles[:idx]
            sep.append(smiles[idx:idx+5])
            smiles = smiles[idx+5:]


        encodings = self.convert_tokens_to_ids(bos + sca + sep + list(smiles) + eos)
        # encodings = self.convert_tokens_to_ids(bos + cls + list(smiles) + eos)
        
        padding_mask = [0] * len(encodings)
        
        if padding and max_length != -1 and len(encodings) < max_length:
            pad_len = (max_length - len(encodings))
            encodings += [self.token2id['[PAD]']] * pad_len
            padding_mask += [1] * pad_len 

        elif max_length != -1 and len(encodings) > max_length:
            encodings = encodings[:max_length]

        return {"input_ids": encodings,
                "padding_mask": padding_mask}

 
    def __call__(self, smiles, padding=False, max_length=-1):
        return self.tokenize(smiles, padding, max_length)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        encodings: List[int] = []
        for char in tokens:
            encodings.append(self.token2id[char])
        return encodings

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
    tokenizer = CharTokenizer('./data/tokenizers/gdb13FullCharTokenizer.json', './data/gdb/gdb13/full')
    
    print(tokenizer.vocab_size)
    smiles = '(CCO)'
    encodings = tokenizer(smiles, max_length=40)
    print(encodings)
    decoded = tokenizer.decode(encodings['input_ids'])
    print(decoded)

if __name__ == "__main__":
    main()
