from unittest import TestCase

import torch

from MolGen.src.models.layers import MultiheadAttention
from MolGen.src.models.transformer import TransformerConfig

class TestMultiHeadAttention(TestCase):
    
    
    def setUp(self) -> None:
        config = TransformerConfig(num_heads=8, block_size=512, proj_dropout_rate=0, attn_dropout_rate=0, n_embd=512, n_layers=2)
        self.attn = MultiheadAttention(config)

        return super().setUp()

    
    def test_attention(self):

        k = torch.tensor([[10, 0, 0],
                        [0, 10, 0],
                        [0, 0, 10],
                        [0, 0, 10]]).float()

        v = torch.tensor([[1, 0],
                            [10, 0],
                            [100, 5],
                            [1000, 6],]).float()

        q = torch.tensor([[0, 0, 10],
                        [0, 10, 0],
                        [10, 10, 0]]).float()

        y, w = self.attn.attention(q, k , v)
        
        expected_output = torch.tensor([[550., 5.5], [10., 0.], [5.5, 0.]])
        eq = torch.isclose(y, expected_output)

        self.assertTrue(torch.all(eq).item())

    def test_mha_shape(self):
        x = torch.rand((1, 60, 512))

        temp = torch.rand((1, 8, 60 ,60))
        y, w = self.attn(x, x, x)
        self.assertTrue(temp.shape == w.shape)
        self.assertTrue(x.shape == y.shape)
