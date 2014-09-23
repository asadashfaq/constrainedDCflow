import unittest
from europe_plusgrid import europe_plus_Nodes

class Test_europe_plus_Nodes(unittest.TestCase):

    def setUp(self):
        self.N = europe_plus_Nodes()

    def test_number_of_countries(self):
        self.assertTrue(len(self.N)==30)

    def test_nonzero_load(self):
        for n in self.N:
            self.assertTrue(n.mean>0)

    def test_alphas_range(self):
        for n in self.N:
            self.assertTrue(n.alpha>=0 and n.alpha <= 1)

if __name__ == '__main__':
    unittest.main()
