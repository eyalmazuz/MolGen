import json
import os
from typing import Dict, List, Union

from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer


class BPETokenizer():

    def __init__(self, tokenizer_path: str='./tokenizers/', data_path: str='./data/', vocab_size: int=500) -> None:


        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self._vocab_size = vocab_size

        print(f'{tokenizer_path=}', f'{os.path.exists(tokenizer_path)=}')
        if tokenizer_path and os.path.exists(tokenizer_path):
            print('Loading Existing tokenizer')
            with open(tokenizer_path, 'r') as f:
                data = json.load(f)
                self.id2token = data['tokens']
                self.merges = {tuple(v): k for k, v in data['merges'].items()}
                self.id2token = {int(k): v for k, v in self.id2token.items()}
        else:
            print('Building tokenizer')
            self.id2token = {}
            self.merges = {}
            if os.path.isdir(data_path):
                for path in os.listdir(data_path):
                    full_path = os.path.join(data_path, path)
                    tokenized_file, merges = self.build_tokenizer(full_path, self._vocab_size)
                    self.id2token, self.merges = {**self.id2token, **tokenized_file}, {**self.merges, **merges}
                    print(self.id2token)
            else:
                self.id2token, self.merges = self.build_tokenizer(data_path, self._vocab_size)

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
                    json.dump({'tokens': self.id2token, 'merges': {v: k for k, v in self.merges.items()}}, f)

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


    def build_tokenizer(self, data_path:str, vocab_size: int) -> Dict[int, str]:
        with open(data_path, 'r') as f:
            molecules = f.readlines()
            corpus = [smiles.strip() for smiles in molecules]


        word_freqs = defaultdict(int)

        for text in corpus:
            words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                word_freqs[word] += 1

        alphabet = []

        for word in word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()

        vocab = alphabet.copy()

        splits = {word: [c for c in word] for word in word_freqs.keys()}

        def compute_pair_freqs(splits):
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                split = splits[word]
                if len(split) == 1:
                    continue
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_freqs[pair] += freq
            return pair_freqs


        pair_freqs = compute_pair_freqs(splits)

        best_pair = ""
        max_freq = None

        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq

        merges = {}

        def merge_pair(a, b, splits):
            for word in word_freqs:
                split = splits[word]
                if len(split) == 1:
                    continue

                i = 0
                while i < len(split) - 1:
                    if split[i] == a and split[i + 1] == b:
                        split = split[:i] + [a + b] + split[i + 2 :]
                    else:
                        i += 1
                splits[word] = split
            return splits

        while len(vocab) < vocab_size:
            pair_freqs = compute_pair_freqs(splits)
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            splits = merge_pair(*best_pair, splits)
            merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])

        id2token = {}
        for i, token in enumerate(vocab):
            id2token[i] = token

        return id2token, merges


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

        pre_tokenize_result = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(smiles)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        splits = [[l for l in word] for word in pre_tokenized_text]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split

        tokens = sum(splits, [])

        encodings = self.convert_tokens_to_ids(bos + sca + sep + tokens + eos)
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
    tokenizer = BPETokenizer('./test.json', './Datasets/gdb13.smi', 500)

    print(tokenizer.vocab_size)
    smiles = '[BOS]C=CC12CC(C#N)C(=O)C1C(=O)O2'
    encodings = tokenizer(smiles, max_length=40, padding=True)
    print(encodings)
    decoded = tokenizer.decode(encodings['input_ids'])
    print(decoded)

if __name__ == "__main__":
    main()
