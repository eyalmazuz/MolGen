import os

from rdkit import Chem
from rdkit.Chem.rdmolfiles import SmilesMolSupplier, MultithreadedSmilesMolSupplier
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from tqdm import tqdm

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

RDLogger.DisableLog('rdApp.*')

def train_tokenizer(data_path: str, save_path: str) -> None:
    """
    Args:
        data_path:
            A path to the smiles data.

        save_path:
            The location to which to save the tokenizer.
    """ 
    print('Training Tokenizer')
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])

    suppl = MultithreadedSmilesMolSupplier(data_path)
    smiles_dataset = []
    for mol in tqdm(suppl):
        if mol:
            smiles = Chem.MolToSmiles(mol)
            smiles_dataset.append(smiles)


    tokenizer.train_from_iterator(smiles_dataset, trainer=trainer)
    print('Saving Tokenizer')
    tokenizer.save(save_path, pretty=True)



def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizer:
    """
    Args:
        tokenizer_path:
            A string of the path the trained tokenizer was saved to.

    Returns: 
        A transformers PreTrainedTokenizer object.
        this tokenizer is used later when training the model.
    """ 
    print('Loading Tokenizer')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    return tokenizer

def main():
    
    data_path = '../../../data/gdb/gdb13/gdb13.rand1M.smi' 
    tokenizer_path =  '../../../data/tokenizers/gdb13BPETokenizer.json'
    if not os.path.exists(tokenizer_path):
        train_tokenizer(data_path, tokenizer_path)

    tokenizer = load_tokenizer(tokenizer_path)
    #tokenizer = Tokenizer.from_file(tokenizer_path)

    #tokenizer.add_special_tokens({"bos_token": '[BOS]', "eos_token":'[EOS]', "pad_token":'[PAD]'})

    print(dir(tokenizer))
    print(help(tokenizer.tokenize))
    smiles = '[BOS]Cc1ncc(C[n+]2csc(CCO)c2C)c(N)n1[EOS]'
    print(f'{smiles=}')
    tokens = tokenizer(smiles)
    print(tokenizer(smiles))
    print(tokenizer.convert_ids_to_tokens(tokens['input_ids']))

if __name__ == "__main__":
    main()
