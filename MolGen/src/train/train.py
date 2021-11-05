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
            
                input_ids = encodings['input_ids'].to(device)
                padding_mask = encodings['padding_mask'].to(device)
                labels = encodings['labels'].to(device)
                
                loss, logits, *args = self.model(input_ids, padding_mask=padding_mask, labels=labels)
                #logits = logits[..., :-1, :]
                #loss = self.criterion(logits.transpose(1, 2), labels)

                
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                loss.backward()
                self.optim.step()

                if batch % 200 == 0:
                    print(f'epoch: {epoch + 1}, batch: {batch + 1}, loss: {loss.item()}')
