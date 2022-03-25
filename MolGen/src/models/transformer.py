import numpy as np
import torch
from torch import nn

from .layers import Encoder, Decoder

class TransformerConfig():
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

class Transoformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        super(Transoformer, self).__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.config = config
        self.register_buffer('mask', 1 - torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))


        self.logits = nn.Linear(config.n_embd, config.vocab_size, bias=False)


    def forward(self, enc_inp, dec_inp, enc_padding_mask=None, dec_padding_mask=None, labels=None, **kwargs):
        B_dec, T_dec = dec_inp.size()
        B_enc, T_enc = enc_inp.size()

        look_ahead_mask = self.mask[:, :, :T_dec, :T_dec]

        if dec_padding_mask is not None:
            attention_mask = dec_padding_mask.view(B_dec, 1, 1, T_dec)
            look_ahead_mask = torch.maximum(look_ahead_mask, attention_mask)

        if enc_padding_mask is not None:
            enc_padding_mask = enc_padding_mask.view(B_enc, 1, 1, T_enc)

        if dec_padding_mask is not None:
            dec_padding_mask = dec_padding_mask.view(B_dec, 1, 1, T_dec)

        enc_out, enc_attnetions = self.encoder(enc_inp, enc_padding_mask)
        
        dec_out, dec_attentions = self.decoder(dec_inp, enc_out, look_ahead_mask, dec_padding_mask)
        
        logits = self.logits(dec_out)

        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
            return loss, logits, {**enc_attnetions, **dec_attentions}
        
        else:
            return logits, {**enc_attnetions, **dec_attentions}

    def generate(self, initial_token, end_token, enc_inp, enc_padding_mask,
                 temprature: int=1, max_len: int=100, device=torch.device('cuda')):
        tokens = [initial_token]
        next_token = -1
        enc_inp = torch.tensor([enc_inp]).to(device)
        enc_padding_mask = torch.tensor([enc_padding_mask]).to(device)
        while next_token != end_token and len(tokens) < max_len:
            x = torch.tensor([tokens]).to(device)
            
            y_pred = self.forward(enc_inp, x, enc_padding_mask=enc_padding_mask)

            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]

            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0)
            if p.device.type != 'cpu':
                p = p.cpu()
            next_token = np.random.choice(len(last_word_logits), p=p.detach().numpy())
            tokens.append(next_token)

        return tokens

    def __str__(self):
        return f"Transformer_Layers_{self.config.n_layers}_Heads_{self.config.num_heads}_Emb_{self.config.n_embd}_Dmodel_{self.config.d_model}"


def main():
    print(5)

if __name__ == "__main__":
    main()
