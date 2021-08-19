from unittest import TestCase

from rdkit import Chem

from MolGen.src.utils.metrics import calc_sas, calc_qed, calc_logp

class LogPTestCase(TestCase):

    def setUp(self):
        self.mol1 = 'c1ccccc1'
        self.mol2 = 'cco'

    def testLogP(self):
        mol1 = Chem.MolFromSmiles(self.mol1)
        self.assertEqual(self.x, foo(self.x))
