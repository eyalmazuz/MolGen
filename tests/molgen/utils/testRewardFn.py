from unittest import TestCase

from MolGen.src.utils.reward_fn import QEDReward

class QEDTestCase(TestCase):

    def setUp(self):
        pass

    def test_object_create(self):
        qed_obj = QEDReward()
        self.assertEqual(0, qed_obj.negative_reward)
        self.assertTrue(callable(qed_obj.multiplier))

    def test_object_create_with_params(self):
        qed_obj = QEDReward(multiplier=lambda x: x, negative_reward=-100)
        self.assertEqual(-100, qed_obj.negative_reward)
        self.assertTrue(callable(qed_obj.multiplier))

    def test_multiplier(self):
        qed_obj = QEDReward()
        res = qed_obj.multiplier(5)
        self.assertEqual(50, res)

    def test_call_with_multiplier(self):
        qed_obj = QEDReward()
        smiles = 'CCC'
        qed = qed_obj(smiles)
        self.assertAlmostEqual(3.854, qed, 2)

    def test_call_without_multiplier(self):
        qed_obj = QEDReward(multiplier=None)
        smiles = 'CCC'
        qed = qed_obj(smiles)
        self.assertAlmostEqual(0.3854, qed, 2)