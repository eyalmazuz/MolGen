from unittest import TestCase

from MolGen.src.models.transformer import Transoformer

class LogPTestCase(TestCase):

    def setUp(self):
        self.x = 5

    def testBuild(self):
        self.assertEqual(self.x, 5)
