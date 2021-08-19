from unittest import TestCase

from MolGen.src.utils.utils import foo

class FooTestCase(TestCase):

    def setUp(self):
        self.x = 5


    def testFoo(self):
        self.assertEqual(self.x, foo(self.x))
