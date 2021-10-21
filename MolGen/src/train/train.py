import torch
from torch.nn.parallel import DistributedDataParallel as DDP

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
                                                 num_workers=8,
                                                 pin_memory=False)

        for epoch in range(epochs):

            for batch, encodings in enumerate(dataloader):
                self.optim.zero_grad()
            
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                labels = encodings['labels'].to(device)
                
                loss, logits, *args = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                #logits = logits[..., :-1, :]
                #loss = self.criterion(logits.transpose(1, 2), labels)


                loss.backward()
                self.optim.step()

                if batch % 200 == 0:
                    #print(self.dataset.decode(x[0].cpu().numpy()), self.dataset.decode(y[0].cpu().numpy()))
                    #print(y.size(), y_pred.size(), y_pred.transpose(1, 2).size())
                    print(f'epoch: {epoch + 1}, batch: {batch + 1}, loss: {loss.item()}')
