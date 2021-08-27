import numpy as np

from rdkit import Chem
from rdkit import RDLogger

import torch

from src.datasets.dataset import CharSmilesDataset
from src.models.recurrent import RecurrentModel
from src.train.train import Trainer
from src.train.evaluate import generate_smiles, get_stats, gen_till_train

RDLogger.DisableLog('rdApp.*')

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CharSmilesDataset('../data/gdb/gdb13/gdb13.rand1M.smi',
                                '../data/tokenizers/gdb13CharTokenizer.json')

    num_embeddings = len(dataset.id2token)
    print(f'{num_embeddings=}')
    print(f'{dataset.max_len=}')
    padding_idx = dataset.token2id['[PAD]']

    embedding_dim = 256
    hidden_size = 64
    num_layers = 2

    model = RecurrentModel(num_embeddings=num_embeddings,
                           embedding_dim=embedding_dim,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           padding_idx=padding_idx).to(device)
    
    optim = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(dataset, model, optim, criterion)
    trainer.train(3, 1000, device)
    
    generated_molecules = generate_smiles(model, dataset, temprature=1)
    get_stats(dataset.molecules, generated_molecules, save_path='../data/results')

    count = gen_till_train(model, dataset)
    print(f'Took {count} Generations')
    
    
if __name__ == "__main__":
    main()
