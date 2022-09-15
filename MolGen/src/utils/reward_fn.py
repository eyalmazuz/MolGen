from abc import ABC, abstractmethod
from collections import OrderedDict

import math
from typing import Callable, Optional, Union, List

import chemprop
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import torch
from tqdm import tqdm

from .metrics import calc_qed, calc_sas

def get_reward_fn(reward_names: List[str], paths: List[str]=None, multipliers: List[str]=None, **kwargs):
    reward_fns = []
    for reward_name, path, mult in zip(reward_names, paths, multipliers):
        if reward_name == 'QED':
            reward_fn = QEDReward(**kwargs)
        
        elif reward_name == 'IC50':
            reward_fn = IC50Reward(**kwargs)

        elif reward_name == 'Anti Cancer':
            reward_fn = ChempropReward(path, reward_name, multiplier=eval(mult))

        elif reward_name == 'LIDI':
            reward_fn = ChempropReward(path, reward_name, multiplier=eval(mult))
        
        reward_fns.append(reward_fn)

    if len(reward_fns) == 1:
        return reward_fns[0]
    else:
        return MultiReward(reward_fns)

class Reward(ABC):
    def __init__(self, multiplier:  Optional[Callable[[float], float]]=None, **kwargs) -> None:
        self.multiplier = multiplier
        
    @abstractmethod
    def __call__(self, smiles: str):
        raise NotImplementedError

class MultiReward(Reward):
    def __init__(self, reward_fns, eval_: bool=False) -> None:
        self.reward_fns = reward_fns

        self._eval = eval_

    def __call__(self, smiles):
        rewards = OrderedDict()
        for fn in self.reward_fns:
            reward = fn(smiles)
            rewards[str(fn)] =  reward

        if not self._eval:
            rewards = list(zip(*list(rewards.values())))
            rewards = [sum(rewards) for rewards in rewards]

        return rewards

    @property
    def eval(self):
        return self._eval

    @eval.setter
    def eval(self, val):
        for fn in self.reward_fns:
            if hasattr(fn, '_eval'):
                fn.eval = val
        self._eval = val

    def __str__(self):
        return "MultiReward"

class ChempropReward(Reward):
    def __init__(self,
                predictor_path,
                name: str='Chemprop',
                multiplier: Optional[Callable[[float], float]]=None,
                eval_: bool=False,
                **kwargs) -> None:
        super().__init__(multiplier=multiplier, **kwargs)
        arguments = [
            '--test_path', '/dev/null',
            '--preds_path', '/dev/null',
            '--checkpoint_dir', predictor_path,
            #'--features_generator', 'rdkit_2d_normalized',
            '--no_features_scaling',
        ]

        self.args = chemprop.args.PredictArgs().parse_args(arguments)

        self.model_objects = chemprop.train.load_model(args=self.args)

        self._eval = eval_
        self.name = name

    def __call__(self, smiles: str) -> float:
        if isinstance(smiles, list):
            smiles = [[s] for s in smiles]
        else:
            smiles = [[smiles]]
        preds = []
        try:
            preds = chemprop.train.make_predictions(args=self.args, smiles=smiles, model_objects=self.model_objects)
            preds = [pred[0] if pred is not None and pred[0] != 'Invalid SMILES' else 0 for pred in preds]
        except TypeError:
            print('Faild to make predictions')
            for s in smiles:
                try:
                    pred = chemprop.train.make_predictions(args=self.args, smiles=[s], model_objects=self.model_objects)
                    preds.append(pred[0] if pred is not None and pred[0] != 'Invalid SMILES' else 0)
                except TypeError:
                    print(f'Bad SMILES: {s[0]}')
                    preds.append(0)

        if self.multiplier is not None and not self.eval:
            #print(preds[:5])
            preds = [self.multiplier(pred) for pred in preds]
            #print(preds[:5])

        return preds

    @property
    def eval(self):
        return self._eval

    @eval.setter
    def eval(self, val):
        self._eval = val

    def __str__(self,):
        return self.name

class IC50Reward(Reward):
    def __init__(self,
                predictor_path,
                tokenizer,
                multiplier: Optional[Callable[[float], float]]=lambda x: math.exp(x / 3),
                **kwargs) -> None:

        super().__init__(multiplier)
        self.predictor = torch.load(predictor_path)
        self.predictor = self.predictor.eval()

        self.tokenizer = tokenizer

    def __call__(self, smiles: str):
        if isinstance(smiles, str):
            smiles = [smiles]
        
        rewards = []
        for s in tqdm(smiles):
            try:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    return 0

                encodings = self.tokenizer('[CLS]' + s, padding=False)

                for k, v in encodings.items():
                    encodings[k] = torch.tensor([v]).long().to(self.predictor.device)
                with torch.no_grad():
                    reward = self.predictor(**encodings)
                if self.multiplier is not None:
                    reward = self.multiplier(reward)
                if isinstance(reward, torch.Tensor):
                    reward = reward.cpu().item()

            except Exception as e:
                # print(f'Failed at: {smiles}')
                reward = 0 

        return rewards

    def __str__(self,):
        return "IC50"
        
class QEDReward(Reward):
    def __init__(self,
                 multiplier: Optional[Callable[[float], float]]=lambda x: x * 10,
                 negative_reward: float=0,
                 **kwargs):

        super(QEDReward, self).__init__(multiplier)
        self.negative_reward = negative_reward
        self.reward_fn = calc_qed

    def __call__(self, smiles: str):
        if isinstance(smiles, str):
            smiles = [smiles]
        
        rewards = []
        for smile in tqdm(smiles):
            try:
                mol = Chem.MolFromSmiles(smile)
                reward = self.reward_fn(mol)
                if self.multiplier:
                    reward = self.multiplier(reward)
            
            except Exception as e:
                reward = self.negative_reward

            rewards.append(reward) 
        
        return rewards

    def __str__(self,):
        return "QED"

def penalized_qed_reward(smiles: Union[List[str], str], fn: Callable[[float], float]=lambda x: x*10) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        qed = calc_qed(mol)
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()

        reward = fn(qed) + 5 * len(rings)# - sum(map(len, rings))
    else:
        reward = 0

    
    return reward

def sas_reward(smiles: str, fn: Callable[[float], float]=lambda x: -math.exp(x/3)) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        sas = calc_sas(mol)
        reward = fn(sas)
    else:
        reward = -1000
    
    return reward

def main():
    with QEDReward() as rfn:
        print(rfn("CCO"))

    with QEDReward() as rfn:
        print(rfn("CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]=]"))

        
if __name__ == "__main__":
    main()
