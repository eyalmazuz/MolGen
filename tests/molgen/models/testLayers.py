from unittest import TestCase

import torch

from MolGen.src.models.layers import Decoder, DecoderBlock, Encoder, MultiheadAttention, EncoderBlock
from MolGen.src.models.transformer import TransformerConfig

class MultiHeadAttentionTestCase(TestCase):
    
    
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

class EncoderBlockTestCase(TestCase):
    
    def setUp(self) -> None:
        config = TransformerConfig(num_heads=8, block_size=512, proj_dropout_rate=0,
                                    attn_dropout_rate=0, n_embd=512, n_layers=2)
        self.enc = EncoderBlock(config)

        return super().setUp()

    def test_encoder_shape(self):
        x = torch.rand((64, 43, 512))

        temp = torch.rand((64, 8, 43 ,43))
        y, w = self.enc(x)
        self.assertTrue(temp.shape == w.shape)
        self.assertTrue(x.shape == y.shape)

class EncoderTestCase(TestCase):
    
    def setUp(self) -> None:
        config = TransformerConfig(num_heads=8, block_size=512, proj_dropout_rate=0,
                                    attn_dropout_rate=0, n_embd=512, n_layers=2)
        self.enc = Encoder(config)

        return super().setUp()

    def test_encoder_shape(self):
        x = torch.randint(0, 512, (64, 62)).long()

        y, _ = self.enc(x)
        self.assertTrue(torch.rand((64, 62 ,512)).shape == y.shape)

class DecoderBlockTestCase(TestCase):
    
    def setUp(self) -> None:
        config = TransformerConfig(num_heads=8, block_size=512, proj_dropout_rate=0,
                                    attn_dropout_rate=0, n_embd=512, n_layers=2)
        self.dec = DecoderBlock(config)
        self.enc = EncoderBlock(config)

        return super().setUp()

    def test_decoder_shape(self):
        enc_inp = torch.rand((64, 43, 512))
        x = torch.rand((64, 50, 512))

        enc_out, _ = self.enc(enc_inp)
        y, _, _ = self.dec(x, enc_out)
        self.assertTrue(x.shape == y.shape)

class DecoderTestCase(TestCase):
    
    def setUp(self) -> None:
        config = TransformerConfig(num_heads=8, block_size=512, proj_dropout_rate=0,
                                    attn_dropout_rate=0, n_embd=512, n_layers=2)
        self.dec = Decoder(config)
        self.enc = Encoder(config)

        return super().setUp()

    def test_decoder_shape(self):
        x = torch.randint(0, 512, (64, 26)).long()
        enc_inp = torch.randint(0, 512, (64, 62)).long()

        enc_out, _ = self.enc(enc_inp)
        y, _ = self.dec(x, enc_out)
        self.assertTrue(torch.rand((64, 26 ,512)).shape == y.shape)