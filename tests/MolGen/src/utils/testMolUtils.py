from unittest import TestCase

from rdkit import Chem

from MolGen.src.utils.mol_utils import get_molecule_scaffold, convert_to_scaffolds, convert_to_molecules, filter_invalid_molecules

class GetScaffoldTestCase(TestCase):

    def setUp(self):
        self.mol1 = 'CCC'
        self.mol2 = 'Cc1cc(Oc2nccc(CCC)c2)ccc1'
        self.mol3 = ''

    def test_have_no_scaffold(self):
        scaffold = get_molecule_scaffold(self.mol1)
        self.assertEqual('', scaffold)

    def test_have_scaffold(self):
        scaffold = get_molecule_scaffold(self.mol2)
        self.assertEqual('c1ccc(Oc2ccccn2)cc1', scaffold)

    def test_invalid_mol(self):
        with self.assertRaises(ValueError):
            get_molecule_scaffold("cccc1")

class ConvertToScaffoldsTestCase(TestCase):

    def setUp(self):
        self.mols1 = ['CCC', 'Cc1cc(Oc2nccc(CCC)c2)ccc1']
        self.mols2 = ['Cc1cc(Oc2nccc(CCC)c2)ccc1', 'Cc1cc(Oc2nccc(CCC)c2)ccc1']
        self.mols3 = []
        self.mols4 = ['CCCAA', 'Cc1cc(Oc2nccc(CCC)c2)ccc1']

    def test_valid_scaffolds(self):
        scaffodls = convert_to_scaffolds(self.mols1)
        self.assertSetEqual({'', 'c1ccc(Oc2ccccn2)cc1'}, scaffodls)
        self.assertEqual(2, len(scaffodls))
    
    def test_repeating_scaffolds(self):
        scaffodls = convert_to_scaffolds(self.mols2)
        self.assertSetEqual({'c1ccc(Oc2ccccn2)cc1'}, scaffodls)
        self.assertEqual(1, len(scaffodls))

    def test_no_mols(self):
        scaffodls = convert_to_scaffolds(self.mols3)
        self.assertSetEqual(set(), scaffodls)
        self.assertEqual(0, len(scaffodls))

    def test_invalid_mol(self):
        with self.assertRaises(ValueError):
            convert_to_scaffolds(self.mols4)


class FilterMolsTestCase(TestCase):

    def setUp(self):
        self.mols1 = ['CCO', 'CCC', 'CCCCC']
        self.mols2 = ['CCA', 'CCO']
        self.mols3 = ['CCCA', 'CCCA', 'CCCA']
        self.mols4 = []

    def test_all_valid(self):
        mols = [Chem.MolFromSmiles(s) for s in self.mols1]
        filtered_mols = filter_invalid_molecules(mols)
        filtered_smiles = [Chem.MolToSmiles(mol) for mol in filtered_mols]
        self.assertListEqual(self.mols1, filtered_smiles)
    
    def test_some_valid(self):
        mols = [Chem.MolFromSmiles(s) for s in self.mols2]
        filtered_mols = filter_invalid_molecules(mols)
        filtered_smiles = [Chem.MolToSmiles(mol) for mol in filtered_mols]
        self.assertListEqual(['CCO'], filtered_smiles)

    def test_none_valid(self):
        mols = [Chem.MolFromSmiles(s) for s in self.mols3]
        filtered_mols = filter_invalid_molecules(mols)
        filtered_smiles = [Chem.MolToSmiles(mol) for mol in filtered_mols]
        self.assertListEqual([], filtered_smiles)

    def test_filter_empty_list(self):
        mols = [Chem.MolFromSmiles(s) for s in self.mols4]
        filtered_mols = filter_invalid_molecules(mols)
        filtered_smiles = [Chem.MolToSmiles(mol) for mol in filtered_mols]
        self.assertListEqual([], filtered_smiles)
    
class ConvertToMolsTestCase(TestCase):

    def setUp(self):
        self.mols1 = ['CCO', 'CCC', 'CCCCC']
        self.mols2 = ['CCA', 'CCO']
        self.mols3 = ['CCCA', 'CCCA', 'CCCA']
        self.mols4 = []

    def test_all_valid(self):
        mols = convert_to_molecules(self.mols1)
        converted_smiles = [Chem.MolToSmiles(mol) for mol in mols]
        self.assertListEqual(self.mols1, converted_smiles)
    
    def test_some_valid(self):
        mols = convert_to_molecules(self.mols2)
        converted_smiles = [Chem.MolToSmiles(mol) if mol is not None else None for mol in mols]
        self.assertListEqual([None, 'CCO'], converted_smiles)

    def test_none_valid(self):
        mols = convert_to_molecules(self.mols3)
        self.assertListEqual([None, None, None], mols)

    def test_conversion_empty_list(self):
        mols = convert_to_molecules(self.mols4)
        converted_smiles = [Chem.MolToSmiles(mol) for mol in mols]
        self.assertListEqual(self.mols4, converted_smiles)