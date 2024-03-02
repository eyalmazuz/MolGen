import json
import os
from unittest import TestCase

from MolGen.src.tokenizers.CharTokenizer import CharTokenizer

class CharTokenizerTestCase(TestCase):

    def setUp(self) -> None:
        with open('./tests/MolGen/src/tokenizers/data.txt', 'w') as f:
            f.write('CCO\n')
            f.write('CCC\n')
            f.write('CCN\n')

        with open('./tests/MolGen/src/tokenizers/et.json', 'w') as f:
            tokens = {"0": "]", "1": "H", "2": "2", "3": "5", "4": ")",
                      "5": "=", "6": "S", "7": "#", "8": "l", "9": "6",
                      "10": "7", "11": "+", "12": "-", "13": "C",
                      "14": "4", "15": "3", "16": "(", "17": "1",
                      "18": "[", "19": "N", "20": "O", "21": "[PAD]",
                      "22": "[BOS]", "23": "[EOS]"}
            json.dump(tokens, f)

    def tearDown(self) -> None:
        if os.path.exists('./tests/MolGen/src/tokenizers/data.txt'):
            os.remove('./tests/MolGen/src/tokenizers/data.txt')
        if os.path.exists('./tests/MolGen/src/tokenizers/tokenizer.json'):    
            os.remove('./tests/MolGen/src/tokenizers/tokenizer.json')
        if os.path.exists('./tests/MolGen/src/tokenizers/et.json'):    
            os.remove('./tests/MolGen/src/tokenizers/et.json')

    def test_create_object(self):
        tokenizer = CharTokenizer('./tests/MolGen/src/tokenizers/tokenizer.json',
                                  './tests/MolGen/src/tokenizers/data.txt')

        self.assertIsNotNone(tokenizer.token2id)
        self.assertIsNotNone(tokenizer.id2token)
        self.assertEqual(7, len(tokenizer.id2token))
        self.assertEqual(7, len(tokenizer.token2id))

    
    def test_create_existing_tokenizer(self):
        tokenizer = CharTokenizer('./tests/MolGen/src/tokenizers/et.json',
                                  data_path=None)

        self.assertIsNotNone(tokenizer.token2id)
        self.assertIsNotNone(tokenizer.id2token)

    def test_tokenize_valid_smile(self):
        smiles = 'CCC'
        tokenizer = CharTokenizer('./tests/MolGen/src/tokenizers/tokenizer.json',
                                  './tests/MolGen/src/tokenizers/data.txt')

        encoding = tokenizer.tokenize(smiles)
        input_ids = encoding['input_ids']
        padding_mask = encoding['padding_mask']

        self.assertTrue(all([isinstance(x, int) for x in input_ids]))
        self.assertListEqual([0] * len(input_ids), padding_mask)
        
        # self.assertEqual(tokenizer.bos_token_id, input_ids[0])
        # self.assertEqual(tokenizer.eos_token_id,/ input_ids[-1])
        self.assertListEqual([tokenizer.token2id['C']] * len(smiles), input_ids)
        
    def test_tokenize_with_bos_eos(self):
        smiles = '[BOS]CCC[EOS]'
        tokenizer = CharTokenizer('./tests/MolGen/src/tokenizers/tokenizer.json',
                                  './tests/MolGen/src/tokenizers/data.txt')

        encoding = tokenizer.tokenize(smiles)
        input_ids = encoding['input_ids']
        padding_mask = encoding['padding_mask']

        self.assertTrue(all([isinstance(x, int) for x in input_ids]))
        self.assertListEqual([0] * len(input_ids), padding_mask)
        
        self.assertEqual(tokenizer.bos_token_id, input_ids[0])
        self.assertEqual(tokenizer.eos_token_id, input_ids[-1])
        self.assertListEqual([tokenizer.bos_token_id] + [tokenizer.token2id['C']] * 3 + [tokenizer.eos_token_id], input_ids)
        
    def test_padding_smiles(self):

        max_len = 10

        smiles = '[BOS]CCC[EOS]'
        tokenizer = CharTokenizer('./tests/MolGen/src/tokenizers/tokenizer.json',
                                  './tests/MolGen/src/tokenizers/data.txt')

        encoding = tokenizer.tokenize(smiles, padding=True, max_length=max_len)
        input_ids = encoding['input_ids']
        padding_mask = encoding['padding_mask']

        self.assertTrue(all([isinstance(x, int) for x in input_ids]))
        self.assertListEqual([0] * 5, padding_mask[:5])
        
        self.assertEqual(tokenizer.bos_token_id, input_ids[0])
        self.assertEqual(tokenizer.eos_token_id, input_ids[4])
        self.assertListEqual([tokenizer.pad_token_id] * (max_len - 5), input_ids[5:])
        self.assertListEqual([1] * (max_len - 5), padding_mask[5:])

        
    def test_convert_tokens_to_ids(self):
        tokens = ['[BOS]', '[EOS]', '[PAD]']
        tokenizer = CharTokenizer('./tests/MolGen/src/tokenizers/tokenizer.json',
                                  './tests/MolGen/src/tokenizers/data.txt')

        
        encodings = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual([tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id], encodings)


    def test_convert_ids_to_tokens(self):
        ids = [0, 0, 4, 5, 3, 3, 3]
        tokenizer = CharTokenizer('./tests/MolGen/src/tokenizers/tokenizer.json',
                                  './tests/MolGen/src/tokenizers/data.txt')

        tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual([tokenizer.id2token[0], tokenizer.id2token[0],
                              '[BOS]', '[EOS]',
                              '[PAD]', '[PAD]', '[PAD]'], tokens)