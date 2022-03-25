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
                
                loss, logits, *_ = self.model(**encodings)
                # output = self.model(**encodings)
                # loss = output['loss']

                loss.backward()
                self.optim.step()

                if batch % 200 == 0:
                    print(f'epoch: {epoch + 1}, batch: {batch + 1}, loss: {loss.item()}')

class PredictorTrainer():
    
    def __init__(self, train_dataset, test_dataset, model, optim, criterion):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.optim = optim
        self.criterion = criterion


    def train(self, epochs, batch_size, device):
        print(f'Train {device}')
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                       shuffle=True,
                                                       batch_size=batch_size,
                                                       num_workers=8,
                                                       collate_fn=self.train_dataset.collate_fn,
                                                       pin_memory=False)

        test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                       shuffle=True,
                                                       batch_size=batch_size,
                                                       num_workers=8,
                                                       collate_fn=self.test_dataset.collate_fn,
                                                       pin_memory=False)

        for epoch in range(epochs):
            self.model = self.model.train()
            for batch, (encodings, labels) in enumerate(train_dataloader):
                self.optim.zero_grad()

                for k, v in encodings.items():
                    encodings[k] = v.to(device)
                labels = labels.to(device).float()

                loss, _ = self.model(**encodings, labels=labels)
                loss.backward()
                self.optim.step()

                if (batch + 1) % 50 == 0:
                    print(f'epoch: {epoch + 1}, batch: {batch + 1}, loss: {loss.item()}')
            
            self.model = self.model.eval()
            
            logits = []
            labels = []
            for batch, (encodings, label) in enumerate(test_dataloader):
                for k, v in encodings.items():
                        encodings[k] = v.to(device)

                with torch.no_grad():
                    preds = self.model(**encodings, test=True)

                logits += preds.cpu().numpy().tolist()
                labels += label.numpy().tolist()
            mse = self.criterion(torch.tensor(logits).float(), torch.tensor(labels).float()) 

            print(f'Epoch: {epoch + 1} Test MSE: {mse}')
