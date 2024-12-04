from rdkit import Chem, DisableLog
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

DisableLog('rdApp.*')



def get_molecule_scaffold(smiles: str) -> str:
    """
    Returns the scaffold of a given molecule.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles)

    return scaffold

def convert_to_scaffolds(mols: list[str]) -> set[str]:
    """
    Returns all the scaffolds that are present in the list of molecules.
    """
    scaffolds = set()
    for mol in tqdm(mols):
        scaffold = get_molecule_scaffold(mol)
        scaffolds.add(scaffold)

    return scaffolds

def convert_to_molecules(smiles_list: list[str]) -> list[Chem.rdchem.Mol]:
    """
    Convert List of SMILES strings to rdkit Mol object.
    """
    mols = [Chem.MolFromSmiles(smiles) for smiles in tqdm(smiles_list)]

    return mols


def filter_invalid_molecules(mols: list[Chem.rdchem.Mol]) -> list[Chem.rdchem.Mol]:
    """
    Filters all the invalid SMILES from the list, and invalid SMIELS is a SMILES that couldn't convert to
    a molecule using rdkit's MolFromSmiles method and the result returned was None.
    """
    mols = list(filter(lambda x: x is not None, mols))

    return mols
