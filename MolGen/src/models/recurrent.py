import torch
from torch import nn


class RecurrentModel(nn.Module):

    def __init__(self,
            num_embeddings,
            embedding_dim,
            hidden_size,
            num_layers,
            padding_idx):

        super(RecurrentModel, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, num_embeddings)


    def forward(self, inputs, attention_mask=None, state=None, labels=None):
        
        embeddings = self.embedding(inputs)
        output, _ = self.lstm(embeddings)
        logits = self.fc(output)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
            return loss, logits

        else:
            return logits

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_size),
                 torch.zeros(self.num_layers, sequence_length, self.hidden_size))

def main():
    model = RecurrentModel(32, 256, 64, 2, 28)
    state_h, state_c = model.init_state(30)

    tens = torch.randint(0, 32, (4, 30))
    print(tens.size())
    y_hat, state = model(tens, (state_h, state_c))

    print(y_hat.size())
    print(y_hat.transpose(1, 2).size())


if __name__ == "__main__":
    main()
