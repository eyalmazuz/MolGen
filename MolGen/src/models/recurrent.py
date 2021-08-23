import torch
from torch import nn


class RecurrentModel(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers):

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_sise = hidden_sise
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=num_ebeddings,
                                      embedding_dim=embedding_dim,
                                      pedding_idx=pedding_idx)

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_sise=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.linear(hidden_size, num_embeddings)


    def forward(self, x, state):
        
        embeddings = self.embedding(x)
        output, state = self.lstm(embeddings, state)
        logits = self.fc(output)

        return logits, state
     def init_state(self, sequence_length):
             return (torch.zeros(self.num_layers, sequence_length, self.hidden_size),
                                     torch.zeros(self.num_layers, sequence_length, self.hidden_size))
 main():
    pass

if __name__ == "__main__":
    main()
