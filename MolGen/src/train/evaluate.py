from typing import List

import numpy as np
from rdkit import Chem
import seaborn as sns
import torch
from tqdm import trange, tqdm

from src.utils.metrics import *
from src.utils.utils import convert_to_mols, filter_invalid_mols

def generate_smiles(model, dataset, temprature=1, size=1000) -> List[Chem.rdchem.Mol]:
    
    model.to('cpu')
    model.eval()
    gen_smiles = []
    for i in trange(size):
        tokens = [dataset.token2id['[BOS]']]
        next_token = ''
        while next_token != dataset.token2id['[EOS]']  and len(tokens) < 36:
            x = torch.tensor([tokens])
            y_pred = model(x)

            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits / temprature, dim=0).detach().numpy()
            next_token = np.random.choice(len(last_word_logits), p=p)
            tokens.append(next_token)

        smiles = dataset.decode(tokens[1:-1])
        gen_smiles.append(smiles)

    #gen_smiles = [Chem.MolFromSmiles(smiles) for smiles in gen_smiles]
    return gen_smiles

def get_stats(train_set, generated_molecules, save_path=None):

    print('Converting smiles to mols')
    train_set = convert_to_mols(train_set)
    generated_molecules = convert_to_mols(generated_molecules)

    print('Calculating percentage of valid mols')
    train_set_valid_count = calc_valid_mols(train_set)
    generated_set_valid_count = calc_valid_mols(generated_molecules)

    print(f'Percentage of valid molecules in the train-set: {train_set_valid_count * 100}')
    print(f'Percentage of valid molecules in the generated-set: {generated_set_valid_count * 100}')

    print('Filtering invlaid mols')
    train_set = filter_invalid_mols(train_set)
    generated_molecules = filter_invalid_mols(generated_molecules)

    # Calculating statics on the train-set.
    print('Calculating Train set stats')

    print('Calculating QED')
    train_qeds = [calc_qed(mol) for mol in tqdm(train_set)]
    plot = sns.kdeplot(train_qeds, color='green', shade=True)
    plot.figure.savefig('./foo.png')

    # Calculating statistics on the generated-set.


def main():
    pass

if __name__ == "__main__":
    main()
