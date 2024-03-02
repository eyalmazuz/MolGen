import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class PredictorConfig():
    def __init__(self,
                vocab_size: int=26,
                n_embd: int=512,
                d_model: int=256,
                n_layers: int=2,
                padding_idx: int=26,
                output_size: int=1,
                **kwargs) -> None:
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.d_model = d_model
        self.n_layers = n_layers
        self.padding_idx = padding_idx
        self.output_size = output_size
        self.kwargs = kwargs

class Predictor(nn.Module):

    def __init__(self, config: PredictorConfig):

        super(Predictor, self).__init__()

        print(f'{config.padding_idx=}')
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.n_embd,
                                      padding_idx=config.padding_idx)

        self.lstm = nn.LSTM(input_size=config.n_embd,
                            hidden_size=config.d_model,
                            # dropout=0.3,
                            num_layers=config.n_layers,
                            batch_first=True)

        # self.dropout = nn.Dropout(p=config.kwargs['proj_dropout_rate'])
        self.fc1 = nn.Linear(config.d_model, config.d_model // 2)
        self.fc2 = nn.Linear(config.d_model // 2, config.output_size)

        self.config = config


    def forward(self, input_ids, **kwargs):
        
        b, t = input_ids.size()
        # if 'test' in kwargs:
            # print(input_ids)
        # print(f'{input_ids.size()=}')
        embeddings = self.embedding(input_ids)
        # print(f'{embeddings.size()=}')
        output, _ = self.lstm(embeddings)
        # print(f'before do {output.size()=}')
        # output = self.dropout(output)
        # print(f'after de {output.size()=}')
        output = output[:, -1, :]
        # print(f'last time step {output.size()=}')
        logits = F.relu(self.fc1(output))
        # print(f'fc1 {logits.size()=}')
        logits = self.fc2(logits)
        # print(f'fc2 {logits.size()=}')
        # print(logits)
        return logits

    @property
    def device(self):
        return next(self.parameters()).device