# MolGen

## Table of Contents

- [Requirements](#requirements)
- [Examples](#examples)
- [Installation](#installation)
- [Data](#data)
- [Tokenizers](#tokenizers)
- [Models](#models)
- [Rewards](#rewards)
- [Training](#training)
    * [Pretraining](#pretraining)
    * [Reinfrocement Learning](#reinforcement-learning)
- [Evaluation](#evaluation)

## Requirements
All requirements are listed in the requirements.txt file

major requirements are
- Rdkit
- Pytorch
- AutoDock VINA

Please refer to the dependencies respective website on installtion information

## Installation

To use this repo it's needed to run it from source
Please clone this repostory using

```
git clone git@github.com:eyalmazuz/MolGen.git
```

For ssh cloning

or

```
git clone https://github.com/eyalmazuz/MolGen.git
```

or simply download by pressing the green button saying ``code`` and then ``Download ZIP``

## Data
The data used in the paper is taken from 3 different sources

### GDB13 dataset rand
This dataset contains a subset of 1 million random molecuels from the larger GDB dataset
access to all GDB datasets can be found in the following link:
https://gdb.unibe.ch/downloads/

### MOSES
This a dataset of lead-like molecuels filtered from the ZINC Clean Leads dataset containing around 1.5 million molecuels
The dataset can be found here: https://github.com/molecularsets/moses

### ZINC 250k
This is a dataset of 250k lead molecules filtered from the large ZINC dataset
Dataset can be found here:
https://www.kaggle.com/datasets/basu369victor/zinc250k


## Tokenizers
Currently the repository supports the following tokenizers:
- Char Tokenizer- Character-level tokenizer that encode each character to a single token
- BPE Tokenizer- Byte-Pair Encoding tokenizer

### Training a Tokenizer
In order to train a tokenizer follow the following steps:

1. Create a directory make tokenizer_data in your favorite location on your machine
2. Open a terminal at the root directory of the project, where the molgen, scripts, etc. folders are located
3. run the following command
```
python3 -m scripts.train_tokenizer arg1 arg2 arg3...
```

#### Command-line Arguments
tokenizer trainign supports the following command-line arguments

* ``--type``- This indicates which types of tokenizer we are training, possible values: BPE, Char
* ``--vocab_size``- This argument is used when training a BPE tokenizer and indicates the final vocabulary (and num of merges - 256) the tokenizer will have
* ``--save_path``- A folder to where to save the trained tokenizer
* ``-data_path``- Path (or paths) to data files used to train the tokenizer, this option supports multiple data files e.g. ``--data_path "/data/to/data/file/one" "/path/to/data/file/two"``
* ``--special_tokens``- Whether to add special tokens to the tokenizer, default value is no special tokens are added. Special tokens are added as list of stings e.g., ``--special_tokens "[BOS]" "[EOS]" "[PAD]" "[UNK]"``

Examples of training tokenizers:
* Char Tokenizer:
```
python3 -m scripts.train_tokenizer --type Char --data_path ./data/smiles.txt --save_path ./data/charTokenizer --special_tokens "<pad>" "<s>" "</s>"
```
* BPE Tokenizer:
```
python3 -m scripts.train_tokenizer --type BPE --data_path ./data/smiles.txt --save_path ./data/bpeTokenizer  --vocab_size 312 --special_tokens "<pad>" "<s>" "</s>"
```

### Creating your own tokenizer
If you wish to use a tokenizer that is currently unsupported by the code, extending the list of tokenizer is fairly simple process.
All tokenizers inherit from AbstractTokenizer that is located at ``molgen/tokenizers/abstract_tokenizer.py``
To create a new tokenizer, simply follow the following steps:

1. Create a new tokenizer file, e.g., ``my_tokenizer.py``
2. Subclass AbstractTokenizer in your new tokenizer class and implement the following 4 methods
    1. encode- This takes a string or a list of strings and return a tokenized values, A list (or tensor) containing ints representing the encoded text
    2. decode- This takes a list of encoded text using int value and returns a string representation of the encoding
    3. load_pretrained- This receives a path containing the tokenizer data and loads the data from the path creating a new class (this is a classmethod)
3. Add the tokenizer class to the tokenizer factory located in ``molgen/tokenizers/tokenizer_factory.py``

##### Notice:
A trained tokenizer must contain a config.json file inside the tokenizer folder, this config file is used to determine which type of tokenizer is loaded.

The config.json file must contain at least the following keys:
* type- The type of tokenizer, this is used in tokenizer factory to decided the class to use
* kwargs (optional)- An inner dict containing arguments that should be passed to the tokenizer init file

The following steps are not required but can be useful:

4. Create a new tokenizer trainer in ``molgen/tokenizer/trainers/``
5. Add the tokenizer as an option to ``--type`` argument in ``molgen/utils/args/tokenizer_args.py``
6. Edit ``molgen/train_tokenizer.py`` to run a trainer for you new tokenizer

## Rewards
Rewards are used during the Reinforcement Learaning phase to optimize our model to generate specific molecules with desired properties. Additionally, it's used during evaluation to evaluate the performance of our models.

### Configuring Rewards

Rewards use a TOML file format to define how to load and choose rewards during our training

Example of a reward TOML config:
```
[rewards.QED]
name = "testQED"

[rewards.QED.scale]
name = "mult"
factor = 10

[rewards.PlogP]
name = "PLOGP"

[rewards.Docking]
name = "Docking BRCA1"
scale = "negate"
```

This TOML config will generate two reward functions one for QED and the other for PlogP.

Since we are using multiple rewards, the code will automatically generate a class called MultiReward that will act as a single reward which is a weighted sum of all the rewards in the TOML config

To define a reward in a TOML format you need to configue the following this
``[rewards.<Reward Name>]``
This will create an entry in the general config file that will use ``<Reward Name>`` class for our reward. Under this we can define two values that can be used in our rewards
1. name- A string-based name for our reward function that is used for debug purposed and human-readable format when evaluating, etc.

2. scale- This will define if you want to apply a numerical transformation to the reward value, scale can be either a string (simiar to name, see Docking in the example config) or a sub key-value entry under ``[rewards.<Reward Name>.scale]`` in our reward dict (see QED example).

    In any case, this will load a use-defined function that is located in ``molgen/rewards/reward_scales.py`` if we choose a dictionary option we can provide extra arguments for the user-defined reward scale.


### Creating a new rewards
Similar to tokenizers, creating new rewards is fairly simple, an AbstractReward class is located in ``molgen/rewards/reward.py`` which all rewards subclass.

To create a new reward simply add to existing py file or create a new py file under ``molgen/rewards/functions/`` and implement the ``__call__`` method of the new class.

``__call__`` need to support receiving either a single SMILES string or a list of SMILES strings, and return a single score or a list of scores respectively

Then, simply add the new reward into the ``name_to_reward`` dictionary located in ``molgen/rewards/reward_factory.py`` and your new reward is ready to be used in the code.


### Creating new user-defined scales
To create a new scale function for any given reward, simply create a new method in ``molgen/rewards/reward_scales.py``, This method needs to support at least receiving the reward score as a parameter and return a single value as their return value.

If additional parameters are needed to the scale, they must follow the value parameter. This is because partial will attempt to put the reward value in the first parameter when calling the function and if there's already a default value given by the user it'll throw an error.

# Running Demo

Please follow the prerequisite before running the code:

1. Create a data folder in root dir of the project,
2. Create a gdb/gdb13 folder and download the GDB13 rand 1m smi file to it from the following link: https://gdb.unibe.ch/downloads/
3. Create a tokenizers folder in the data folder.
4. Create a results folder in the data folder.

All the code contains the hyper-parameters used in all of the expremiments

To train a language model and then perform reinforcement learning optimization run:
``python3 MolGen/main.py --do_train --do_eval --dataset_path ./data/gdb/gdb13/gdb13.smi --tokenizer Char --tokenizer_path ./data/tokenizers/gdb13CharTokenizer.json --reward_fns QED --multipliers "lambda x: x" --batch_size 256``

To only perform reinfocement learning optimization with a pretrained language model run:
``python3 MolGen/main.py --load_pretrained --pretrained_path ./data/models/gpt_pre_rl_gdb13.pt --do_eval --dataset_path ./data/gdb/gdb13/gdb13.smi --tokenizer Char --tokenizer_path ./data/tokenizers/gdb13CharTokenizer.json --reward_fns QED --multipliers "lambda x: x" --batch_size 256``

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
