import torch
from torch import nn

class RecurrentConfig():
    def __init__(self,
                vocab_size=26,
                n_embd=512,
                d_model=256,
                n_layers=2,
                padding_idx=26,
                **kwargs) -> None:
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.d_model = d_model
        self.n_layers = n_layers
        self.padding_idx = padding_idx

class RecurrentModel(nn.Module):

    def __init__(self, config):

        super(RecurrentModel, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.n_embd,
                                      padding_idx=config.padding_idx)

        self.lstm = nn.LSTM(input_size=config.n_embd,
                            hidden_size=config.d_model,
                            num_layers=config.n_layers,
                            batch_first=True)

        self.fc = nn.Linear(config.d_model, config.vocab_size)


    def forward(self, inputs, attention_mask=None, state=None, labels=None):
        
        embeddings = self.embedding(inputs)
        output, _ = self.lstm(embeddings)
        logits = self.fc(output)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
            return loss, logits

        else:
            return logits

def main():
    config = RecurrentConfig(padding_idx=4)
    model = RecurrentModel(config)

    tens = torch.randint(0, 4, (1, 4))
    print(tens.size())
    y_hat = model(tens)

    print(y_hat.size())
    print(y_hat)
    print(y_hat.argmax(2))
    print(y_hat[0].gather(1, y_hat.argmax(2).view(-1 ,1)))
    print(y_hat.transpose(1, 2).size())


if __name__ == "__main__":
    main()
