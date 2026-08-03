import os
import sys

import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP  # type: ignore
from rdkit.Chem.QED import qed
from rdkit.Chem.rdchem import Mol
from rdkit.RDConfig import RDContribDir
sys.path.append(os.path.join(RDContribDir, 'SA_Score'))
import sascorer

from molgen.rewards.reward import AbstractReward, RewardScale


class QEDReward(AbstractReward):
    def __init__(self, name: str | None = None, scale: RewardScale = None) -> None:
        super().__init__(name=name, scale=scale)

    def __call__(self, smiles: str | list[str]) -> float | list[float]:
        if isinstance(smiles, str):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0
            else:
                reward = qed(mol)
                if self.scale is not None and not self.eval:
                    reward = self.scale(reward)

                return reward

        else:
            mols = [Chem.MolFromSmiles(s) for s in smiles]
            rewards = [qed(mol) if mol is not None else -1 for mol in mols]

            if self.scale is not None and not self.eval:
                rewards = [self.scale(reward) for reward in rewards]

            return rewards


class PenalizedLogPReward(AbstractReward):
    def __init__(self, name: str | None = None, scale: RewardScale = None) -> None:
        super().__init__(name=name, scale=scale)

    def __call__(self, smiles: str | list[str]) -> float | list[float]:
        if isinstance(smiles, str):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0
            else:
                reward = PenalizedLogPReward.penalized_logp(mol)
                if self.scale is not None and not self.eval:
                    reward = self.scale(reward)

                return reward

        else:
            mols = [Chem.MolFromSmiles(s) for s in smiles]
            rewards = [PenalizedLogPReward.penalized_logp(mol) if mol is not None else -1 for mol in mols]

            if self.scale is not None and not self.eval:
                rewards = [self.scale(reward) for reward in rewards]

            return rewards

        pass

    @staticmethod
    def num_long_cycles(mol: Mol) -> int:
        """Calculate the number of long cycles.

        Args:
          mol: Molecule. A molecule.

        Returns:
          negative cycle length.
        """
        cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
        cycle_length = 0 if not cycle_list else max([len(j) for j in cycle_list])
        cycle_length = 0 if cycle_length <= 6 else cycle_length - 6
        return cycle_length

    @staticmethod
    def penalized_logp(molecule: Mol) -> float:
        log_p = MolLogP(molecule)
        sas_score = sascorer.calculateScore(molecule)
        cycle_score = PenalizedLogPReward.num_long_cycles(molecule)
        if sas_score is None:
            smiles = Chem.MolToSmiles(molecule) if molecule is not None else "<none>"
            raise ValueError(f"SAS scorer returned None for molecule: {smiles}")
        return log_p - sas_score - cycle_score


class pIC50Reward(AbstractReward):
    def __init__(self,
                 data_path: str,
                 name: str | None = None,
                 scale: RewardScale | None = None) -> None:
        super(pIC50Reward, self).__init__(name=name, scale=scale)
        df = pd.read_csv(data_path)
        smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in df['smiles'] if Chem.MolFromSmiles is not None]
        self.smiles_to_pIC50 = dict(zip(smiles, df['KRAS pIC50']))

    def __call__(self, smiles: str | list[str]) -> float | list[float]:
        return self.smiles_to_pIC50[smiles]
