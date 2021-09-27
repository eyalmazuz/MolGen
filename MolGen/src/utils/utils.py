import os

import matplotlib.pyplot as plt

import seaborn as sns
from tqdm import tqdm

def generate_and_save_plot(values,
                           plot_func,
                           xlabel,
                           ylabel,
                           title,
                           save_path,
                           name,
                           **kwargs):
    
    plot = plot_func(values, **kwargs)
    plot.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plot.figure.savefig(f'{save_path}/{name}.png')
    plt.clf()


def get_max_smiles_len(data_path: str) -> int:
    if os.path.isdir(data_path):
        max_len = 0
        for path in os.listdir(data_path):
            full_path = os.path.join(data_path, path)
            file_max_len = len(max(open(full_path, 'r'), key=len))
            max_len = file_max_len if file_max_len > max_len else max_len
    else:
        max_len = len(max(open(data_path, 'r'), key=len))
    
    return max_len + 2