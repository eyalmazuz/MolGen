from unittest import TestCase

import torch

from MolGen.src.models.transformer import Transoformer, TransformerConfig

class TransformerTestCase(TestCase):
    
    def setUp(self) -> None:
        config = TransformerConfig(num_heads=8, block_size=512, proj_dropout_rate=0,
                                    attn_dropout_rate=0, n_embd=512, n_layers=2)
        self.transformer = Transoformer(config)

        return super().setUp()

    def test_transformer_shape(self):
        enc_inp = torch.randint(0, 512, (64, 38)).long()
        dec_inp = torch.randint(0, 512, (64, 36)).long()

        y, _ = self.transformer(enc_inp, dec_inp)
        self.assertTrue(torch.rand((64, 36 ,512)).shape == y.shape)