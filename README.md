# MolGen

# Overview
this code is used for Molecule Generation Using Transformers and Policy Gradient Reinfocement Learning

# System Requirements
the code ran on a 8-core CPU with 64GB or ram and TITAN RTX GPU.
using Linux: Ubuntu 18.04+

# Installtion Guide
Install the conda enviroment using the following command:
```conda env create -f environment.yml```

# Running Demo

Please follow the prerequisite before running the code:

1. Create a data folder in root dir of the project,
2. Create a gdb/gdb13 folder and download the GDB13 rand 1m smi file to it from the following link: https://gdb.unibe.ch/downloads/
3. Create a tokenizers folder in the data folder.
4. Create a results folder in the data folder.

All the code contains the hyper-parameters used in all of the expremiments
so to reproduce the results simply run the following command
```python3 MolGen/main.py```

# Cite
Mazuz, E., Shtar, G., Shapira, B. et al. Molecule generation using transformers and policy gradient reinforcement learning. Sci Rep 13, 8799 (2023). https://doi.org/10.1038/s41598-023-35648-w

```
@article{mazuz2023molecule,
  title={Molecule generation using transformers and policy gradient reinforcement learning},
  author={Mazuz, Eyal and Shtar, Guy and Shapira, Bracha and Rokach, Lior},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={8799},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
