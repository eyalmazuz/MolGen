import numpy as np
import torch
from torch import nn

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

class Predictor(nn.Module):

    def __init__(self, config: PredictorConfig):

        super(Predictor, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.n_embd,
                                      padding_idx=config.padding_idx)

        self.lstm = nn.LSTM(input_size=config.n_embd,
                            hidden_size=config.d_model,
                            num_layers=config.n_layers,
                            batch_first=True)

        self.fc = nn.Linear(config.d_model, config.output_size)

        self.config = config


    def forward(self, input_ids, **kawrgs):
        
        embeddings = self.embedding(input_ids)
        output, _ = self.lstm(embeddings)
        logits = self.fc(output[:, -1, :])

        return logits