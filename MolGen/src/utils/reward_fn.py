from abc import ABC, abstractmethod

import math
from typing import Callable, Optional

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from .metrics import calc_qed, calc_sas


def get_reward_fn(reward_name: str='QED', **kwargs):

    if reward_name == 'QED':
        return QEDReward(**kwargs)
    
    elif reward_name == 'IC50':
        return IC50Reward(**kwargs)

class Reward(ABC):

    def __init__(self, multiplier:  Optional[Callable[[float], float]]=None, **kwargs) -> None:
        self.multiplier = multiplier
        
    @abstractmethod
    def __call__(self, smiles: str):
        raise NotImplementedError

class IC50Reward(Reward):

    def __init__(self,
                predictor,
                multiplier: Optional[Callable[[float], float]]=None,
                **kwargs) -> None:

        super().__init__(multiplier)
        self.predictor = predictor

    def __call__(self, smiles: str):
        predicted_ic50 = self.predictor(smiles)
        reward = self.multiplier(predicted_ic50)
        return reward
        
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
        try:
            mol = Chem.MolFromSmiles(smiles)
            reward = self.reward_fn(mol)
            if self.multiplier:
                return self.multiplier(reward)
            else:
                return self.reward_fn(mol)
        
        except Exception as e:
            return self.negative_reward
        
    def __str__(self,):
        return "QED"

def penalized_qed_reward(smiles: str, fn: Callable[[float], float]=lambda x: x*10) -> float:

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