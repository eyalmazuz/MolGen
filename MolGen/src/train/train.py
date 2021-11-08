import os

import torch

class Trainer():
    
    def __init__(self, dataset, model, optim, criterion):
        self.dataset = dataset
        self.model = model
        self.optim = optim
        self.criterion = criterion


    def train(self, epochs, batch_size, device):
        print(f'Train {device}')
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=8,
                                                 pin_memory=False)

        for epoch in range(epochs):

            for batch, encodings in enumerate(dataloader):
                self.optim.zero_grad()

                for k, v in encodings.items():
                    encodings[k] = v.to(device)
                
                loss, logits, *args = self.model(**encodings)

                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                loss.backward()
                self.optim.step()

                if batch % 200 == 0:
                    print(f'epoch: {epoch + 1}, batch: {batch + 1}, loss: {loss.item()}')
