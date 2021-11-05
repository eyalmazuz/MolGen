import os
from unittest import TestCase

from rdkit import Chem

from MolGen.src.utils.utils import get_max_smiles_len

class MaxLenTestCase(TestCase):

    def setUp(self):

        with open('./tests/MolGen/src/utils/test_mols1.txt', 'w') as f:
            f.write('CCO\n')
            f.write('CCO\n')
        
        with open('./tests/MolGen/src/utils/test_mols2.txt', 'w') as f:
            f.write('CCO\n')
            f.write('CCCO\n')
        
        with open('./tests/MolGen/src/utils/test_mols3.txt', 'w') as f:
            pass

    
    def tearDown(self) -> None:
       os.remove('./tests/MolGen/src/utils/test_mols1.txt')
       os.remove('./tests/MolGen/src/utils/test_mols2.txt')
       os.remove('./tests/MolGen/src/utils/test_mols3.txt') 


    def test_same_length(self):
        max_len = get_max_smiles_len('./tests/MolGen/src/utils/test_mols1.txt')
        self.assertEqual(3, max_len)

    def test_different_length(self):
        max_len = get_max_smiles_len('./tests/MolGen/src/utils/test_mols2.txt')
        self.assertEqual(4, max_len)

    def test_no_smiles(self):
        with self.assertRaises(ValueError):
            max_len = get_max_smiles_len('./tests/MolGen/src/utils/test_mols3.txt')