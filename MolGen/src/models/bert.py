import numpy as np
import torch
from torch import nn

from .layers import Encoder, Decoder

class BertConfig():
    def __init__(self,
                vocab_size=512,
                n_embd=512,
                block_size=512,
                proj_size=512,
                d_model=512,
                num_heads=8,
                n_layers=12,
                attn_dropout_rate=0.1,
                proj_dropout_rate=0.1,
                resid_dropout_rate=0.1,
                embd_dropout_rate=0.1,
                **kwargs
                ) -> None:
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.proj_size = proj_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.attn_dropout_rate = attn_dropout_rate
        self.proj_dropout_rate = proj_dropout_rate
        self.resid_dropout_rate = resid_dropout_rate
        self.embd_dropout_rate = embd_dropout_rate

class Bert(nn.Module):

    def __init__(self, config: BertConfig):
        super(Bert, self).__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.config = config

        self.logits = nn.Linear(config.n_embd, 1, bias=True)


    def forward(self, input_ids, padding_mask=None, labels=None, **kwargs):
        B_enc, T_enc = input_ids.size()

        if padding_mask is not None:
            padding_mask = padding_mask.view(B_enc, 1, 1, T_enc)

        enc_out, enc_attnetions = self.encoder(input_ids, padding_mask)
        
        # print(f'{enc_out.size()=}')
        pool = enc_out[:, 0, :]
        # print(f'{pool.size()=}')
        logits = self.logits(pool)
        # print(f'{logits.size()=}')

        
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
            return loss, logits #, enc_attnetions
        
        else:
            return logits #, enc_attnetions

    def __str__(self):
        return f"Bert_Layers_{self.config.n_layers}_Heads_{self.config.num_heads}_Emb_{self.config.n_embd}_Dmodel_{self.config.d_model}"

    @property
    def device(self):
        return next(self.parameters()).device

def main():
    print(5)

if __name__ == "__main__":
    main()
