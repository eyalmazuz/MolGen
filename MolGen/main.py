import numpy as np
import os

from rdkit import Chem
from rdkit import RDLogger

import torch
from torch.utils.data import ConcatDataset

from src.datasets.dataset import get_dataset
from src.models.model_builder import get_model, ModelOpt
from src.tokenizers.CharTokenizer import CharTokenizer
from src.train.train import Trainer
from src.train.evaluate import generate_smiles, get_stats, gen_till_train
from src.train.reinforcement import policy_gradients
from src.utils.metrics import calc_qed
from src.utils.utils import get_max_smiles_len

RDLogger.DisableLog('rdApp.*')
torch.autograd.set_detect_anomaly(True)
def main():

    data_path = './data/gdb/gdb13/gdb13.rand1M.smi'
    tokenizer_path = './data/tokenizers/gdb13CharTokenizer.json'

    max_len = get_max_smiles_len(data_path)
    print(max_len)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = CharTokenizer(tokenizer_path)
    dataset = get_dataset(data_path,
                          tokenizer=tokenizer,
                          max_len=max_len,
                          to_load=True)

    config = {
        'n_emb': 64,
        'd_model': 64,
        'n_layers': 1,
        'num_heads': 8,
        'vocab_size': tokenizer.vocab_size,
        'n_positions': 512,
        'proj_size': 512,
        'attn_dropout_rate': 0.1,
        'proj_dropout_rate': 0.1,
        'resid_dropout_rate': 0.1,
        'padding_idx': tokenizer.pad_token_id

    }

    model = get_model(ModelOpt.RECURRENT, **config).to(device)

    optim = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(dataset, model, optim, criterion)
    trainer.train(5, 1024, device)

    # generated_molecules = generate_smiles(model, tokenizer, temprature=1)
    # get_stats(dataset.molecules, generated_molecules, save_path='../data/results')

    policy_gradients(model, tokenizer, reward_fn=calc_qed, batch_size=100, epochs=25, discount_factor=0.99)
    generated_molecules = generate_smiles(model, tokenizer, temprature=1, size=1000)
    get_stats(data_path, generated_molecules, save_path='./data/results')

    #count = gen_till_train(model, dataset)
    #print(f'Took {count} Generations for generate a mol from the test set.')
    
if __name__ == "__main__":
    main()
