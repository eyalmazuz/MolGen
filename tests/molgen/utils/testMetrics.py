from unittest import TestCase

from rdkit import Chem

from MolGen.src.utils.metrics import calc_novelty, calc_diversity, calc_valid_molecules

class DiversityTestCase(TestCase):

    def setUp(self):
        self.gen_mols1 = ['CC', 'CCO']
        self.gen_mols2 = ['CCC', 'CCC', 'CCC']
        self.gen_mols3 = ['CCC', 'CCC', 'CC']
        self.gen_mols4 = []

    def test_diversity_all_different(self):
        diversity_score = calc_diversity(self.gen_mols1)
        self.assertEqual(diversity_score, 1)

    def test_diversity_all_same(self):
        diversity_score = calc_diversity(self.gen_mols2)
        self.assertAlmostEqual(diversity_score, 0.333, 2)

    def test_diversity_some_different(self):
        diversity_score = calc_diversity(self.gen_mols3)
        self.assertAlmostEqual(diversity_score, 0.666, 2)

    def test_diversity_empty_list(self):
        with self.assertRaises(ZeroDivisionError):
            calc_diversity(self.gen_mols4)

class NoveltyTestCase(TestCase):

    def setUp(self):
        self.train_mol = ['CCO', 'CCC', 'CCCCC']
        self.gen_mols1 = ['CC', 'CCO']
        self.gen_mols2 = ['CCC', 'CCC', 'CCC']
        self.gen_mols3 = ['CCCCCCCC']
        self.gen_mols4 = []

    def test_novelty_all_different(self):
        novelty_score = calc_novelty(self.train_mol, self.gen_mols1)
        self.assertEqual(novelty_score, 0.5)
    
    def test_novelty_all_same(self):
        novelty_score = calc_novelty(self.train_mol, self.gen_mols2)
        self.assertEqual(novelty_score, 0)

    def test_novelty_some_different(self):
        novelty_score = calc_novelty(self.train_mol, self.gen_mols3)
        self.assertAlmostEqual(novelty_score, 1, 2)

    def test_novelty_empty_list(self):
        with self.assertRaises(ZeroDivisionError):
            calc_novelty(self.train_mol, self.gen_mols4)

class ValidityTestCase(TestCase):

    def setUp(self):
        self.gen_mols1 = ['CCO', 'CCC', 'CCCCC']
        self.gen_mols2 = ['CCA', 'CCO']
        self.gen_mols3 = ['CCCA', 'CCCA', 'CCCA']
        self.gen_mols4 = []

    def test_all_valid(self):
        validity_score = calc_valid_molecules(self.gen_mols1)
        self.assertEqual(validity_score, 1)
    
    def test_some_valid(self):
        validity_score = calc_valid_molecules(self.gen_mols2)
        self.assertEqual(validity_score, 0.5)

    def test_none_valid(self):
        validity_score = calc_valid_molecules(self.gen_mols3)
        self.assertEqual(validity_score, 0)

    def test_validity_empty_list(self):
        with self.assertRaises(ZeroDivisionError):
            calc_valid_molecules(self.gen_mols4)