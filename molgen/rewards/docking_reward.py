from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from rdkit import Chem

from src.rewards.reward import AbstractReward

# TODO: consider changing docking to use pydock or something more simple
class DockingReward(AbstractReward):
    def __init__(self,
                 receptor_path,
                 name: Optional[str]=None,
                 scale: Optional[Callable[[float], float]]=None,
                 center: Optional[Tuple[float, float, float]]=None,
                 size: Optional[Tuple[float, float, float]]=None) -> None:

        super(DockingReward, self).__init__(name=name, scale=scale)

        self.receptor_path = receptor_path
        if center is None: 
            protein = Chem.MolFromPDBFile(receptor_path[:-2]) #we remove the last 2 chars with will resutls in reading the PDB file
            pos = protein.GetConformer(0).GetPositions()
            self.center = (pos.max(0) + pos.min(0)) / 2
            print(f"Protein center is: {self.center}")
        else:
            self.center = center

        self.size = size
        self.vina = Vina(sf_name='vina', cpu=0, verbosity=1)

    def __call__(self, smiles: Union[str, List[str]]) -> Union[float, List[float]]:
        if isinstance(smiles, str):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0
            else:
                return self.__dock(s)
        else:
            rewards = [self.__dock(s) if Chem.MolFromSmiles(s) is not None else 0 for s in smiles]
            if self.scale is not None and not self.eval:
                rewards = [self.scale(reward) for reward in rewards]
            return rewards

    def __dock(self, smiles: str) -> float:
        try:
            # Create RDKit molecule object
            mol = Chem.MolFromSmiles(smiles)
            mol = AllChem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if mol.GetNumConformers() > 0:
                AllChem.MMFFOptimizeMolecule(mol)

            else:
                return 0

            # Prepare mol
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            for setup in mol_setups:
                pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)


            if not is_ok:
                return 0
            # with open(f"./data/proteins/{smiles}.pdbqt", 'w') as f:
            #    f.write(pdbqt_string)
        
            # Configure Vina
            self.vina.set_receptor(self.receptor_path)
            self.vina.set_ligand_from_string(pdbqt_string)

            # Define the search space (coordinates and dimensions)
            x, y, z = self.center
            if self.box_size is None:
                size_x = size_y = size_z = 30
            else:
                size_x, size_y, size_z = self.size

            self.vina.compute_vina_maps(center=[x, y, z], box_size=[size_x, size_y, size_z])

            # Run docking
            self.vina.dock(n_poses=5, exhaustiveness=32)

            score = self.vina.score()[0]

            return score

        except Exception:
            return 0
    
    def __str__(self) -> str:
        protein_name = Path(self.receptor_path).stem
        return f"{protein_name} Docking {self.name}"
