import torch


class Trainer():
    
    def __init__(self, dataset, model, optim, criterion):
        self.dataset = dataset
        self.model = model
        self.optim = optim
        self.criterion = criterion


    def train(self, epochs, batch_size, device):
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=8)

        for epoch in range(epochs):

            for batch, (x, y) in enumerate(dataloader):
                self.optim.zero_grad()
            
                x = x.to(device)
                y = y.to(device)

                y_pred = self.model(x)

                loss = self.criterion(y_pred.transpose(1, 2), y)


                loss.backward()
                self.optim.step()

                if batch % 200 == 0:
                    #print(self.dataset.decode(x[0].cpu().numpy()), self.dataset.decode(y[0].cpu().numpy()))
                    #print(y.size(), y_pred.size(), y_pred.transpose(1, 2).size())
                    print(f'epoch: {epoch + 1}, batch: {batch + 1}, loss: {loss.item()}')
        
