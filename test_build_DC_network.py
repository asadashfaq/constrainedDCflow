import unittest
from europe_plusgrid import europe_plus_Nodes
import aurespf.solvers as au
import DCsolvers as DCs

class Test_build_DC_network(unittest.TestCase):

    def setUp(self):
        self.N = europe_plus_Nodes()
        self.model = DCs.build_DC_network(self.N)[0]
        self.Nlinks = len(au.AtoKh(self.N)[-1])
        self.Nnodes = len(self.N)

    def test_number_of_vars(self):
        expected_number = 3*self.Nnodes
        self.assertTrue(len(self.model.getVars())==expected_number)

    def test_number_of_constrs(self):
        expected_number = self.Nnodes + 1 + 2*self.Nlinks
        self.assertTrue(len(self.model.getConstrs())==expected_number)


if __name__ == '__main__':
    unittest.main()
