import numpy as np

from rdkit import Chem
from rdkit import RDLogger

import torch

from src.datasets.dataset import SmilesDataset
from src.train.evaluate import generate_smiles, get_stats, gen_till_train
from src.models.recurrent import RecurrentModel
from src.tokenizers.CharTokenizer import CharTokenizer
from src.train.train import Trainer

RDLogger.DisableLog('rdApp.*')

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = CharTokenizer('../data/tokenizers/gdb13CharTokenizer.json')
    dataset = SmilesDataset('../data/gdb/gdb13/gdb13.rand1M.smi',
                            tokenizer)

    vocab_size = tokenizer.vocab_size
    print(f'{vocab_size=}')


    smiles = 'CC1C(C)C(C(CC#N)C=C)C1C'
    print(f'{dataset.max_len=}')
    padding_idx = tokenizer.pad_token_id

    embedding_dim = 256
    hidden_size = 64
    num_layers = 2

    model = RecurrentModel(num_embeddings=vocab_size,
                           embedding_dim=embedding_dim,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           padding_idx=padding_idx).to(device)

    optim = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(dataset, model, optim, criterion)
    trainer.train(3, 1024, device)

    generated_molecules = generate_smiles(model, tokenizer, temprature=1)
    print(generated_molecules)
    get_stats(dataset.molecules, generated_molecules, save_path='../data/results')

    count = gen_till_train(model, dataset, type='mol')
    print(f'Took {count} Generations for generate a mol from the dataset.')
    
    count = gen_till_train(model, dataset, type='scaffold')
    print(f'Took {count} Generations for generate a scaffold from the dataset.')
    
    
if __name__ == "__main__":
    main()
