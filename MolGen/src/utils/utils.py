import os
from typing import List

import matplotlib.pyplot as plt

import seaborn as sns
from tqdm import tqdm

def generate_and_save_plot(values: List[float],
                           plot_func,
                           xlabel: str,
                           ylabel: str,
                           title: str,
                           save_path: str,
                           name: str,
                           **kwargs) -> None:
    """
    Generates a plot and saves it.
    """
    
    plot = plot_func(values, **kwargs)
    plot.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plot.figure.savefig(f'{save_path}/{name}.png')
    plt.clf()


def get_max_smiles_len(data_path: str) -> int:
    """
    Returns the length of the molecule which has the longest SMILES string.
    this is used for padding in the tokenizer when traning the model.  
    """
    if os.path.isdir(data_path):
        max_len = 0
        for path in os.listdir(data_path):
            full_path = os.path.join(data_path, path)
            file_max_len = len(max(open(full_path, 'r'), key=len))
            max_len = file_max_len if file_max_len > max_len else max_len
    else:
        max_len = len(max(open(data_path, 'r'), key=len))
    
    return max_len + 2