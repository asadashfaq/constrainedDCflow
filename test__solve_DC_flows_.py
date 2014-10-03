import unittest
from europe_plusgrid import europe_plus_Nodes
import aurespf.solvers as au
import DCsolvers as DCs

class Test__solve_DC_flows_(unittest.TestCase):

    def setUp(self):
        self.N = europe_plus_Nodes()
        self.mean_loads = [n.mean for n in self.N]
        self.mode = 'linear'
        self.network_constr, self.sum_of_squared_flows_constr \
                                     = DCs.build_DC_network(self.N, b=1e3)
        self.network_copper, self.sum_of_squared_flows_copper\
                                  = DCs.build_DC_network(self.N, copper=True)
        self.Nlinks = len(au.AtoKh(self.N)[-1])
        self.Nnodes = len(self.N)
        self.testtimesteps = [52094, 52095, 52096, \
                              0, 7, 50, 100, 1434, 32049, 198391]

    def test_number_of_vars(self):
        for t in self.testtimesteps:
            DCs._solve_DC_flows_(self.network_constr, self.N, t,\
                                 self.mode, self.sum_of_squared_flows_constr,
                                 self.mean_loads)
            expected_number = 3*self.Nnodes
            self.assertTrue(len(self.network_constr.getVars())==expected_number)


    def test_number_of_constrs(self):
        for t in self.testtimesteps:
            DCs._solve_DC_flows_(self.network_constr, self.N, t, \
                                 self.mode, self.sum_of_squared_flows_constr,\
                                 self.mean_loads)
            expected_number = self.Nnodes + 1 + 2*self.Nlinks
            self.assertTrue(\
                    len(self.network_constr.getConstrs())==expected_number)

    def test_only_bal_or_curt(self):
        """ This test makes sure that in the unconstrained case,
            all the total balancing or the total curtailment is 0

            """

        for t in self.testtimesteps:
            results = DCs._solve_DC_flows_(self.network_copper, self.N, t, \
                                  self.mode, self.sum_of_squared_flows_copper,\
                                  self.mean_loads)
            bal = results[1::3]
            curt = results[2::3]
            self.assertTrue(sum(bal)<=1e-5 or sum(curt)<=1e-5)


if __name__ == '__main__':
    unittest.main()

