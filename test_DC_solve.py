import unittest
import numpy as np
from europe_plusgrid import europe_plus_Nodes
import aurespf.solvers as au
import DCsolvers as DCs

class Test_DC_solve(unittest.TestCase):

    def setUp(self):
        self.N = europe_plus_Nodes()
        self.testlapse = range(200)
        self.h0_sqr = au.get_quant_caps(\
                filename='./results/Europe_aHE_copper_sqr_flows.npy')
        self.h0_lin = au.get_quant_caps(\
                        filename='./results/Europe_aHE_copper_lin_flows.npy')

    def test_constrained_h0_flow(self):
        print "Test constrained_flow h0"
        Nsolved, F = DCs.DC_solve(self.N, mode='linear',\
                            h0=self.h0_lin, b=0.1,
                            msg='Solving constrained lin flows',
                            lapse=self.testlapse)

        for t in self.testlapse:
            self.assertTrue(sum([n.balancing[t] + n.curtailment[t]\
                                 for n in Nsolved]) > 0.1)

    def test_unconstrained_lin_flow(self):
        print "Test unconstrained lin flow "
        Nsolved, F = DCs.DC_solve(self.N, mode='linear copper',\
                            msg='Solving unconstrained lin flows',
                            lapse=self.testlapse)

        for t in self.testlapse:
            self.assertTrue(sum([n.balancing[t] + n.curtailment[t]\
                                 for n in Nsolved]) > 0.0)

    def test_unconstrained__sqr_flow(self):
        print "Test unconstrained sqr flow "
        Nsolved, F = DCs.DC_solve(self.N, mode='square copper',\
                            msg='Solving unconstrained sqr flows',
                            lapse=self.testlapse)
        for t in self.testlapse:
            self.assertTrue(sum([n.balancing[t] + n.curtailment[t]\
                                 for n in Nsolved]) > 0.0)

    def test_constrained_h0_sqr_flow(self):
        print "Test constrained_flow h0, sqr"
        Nsolved, F = DCs.DC_solve(self.N, mode='square',\
                            h0=self.h0_sqr, b=0.1,
                            msg='Solving constrained sqr flows',
                            lapse=self.testlapse)

        for t in self.testlapse:
            self.assertTrue(sum([n.balancing[t] + n.curtailment[t]\
                                 for n in Nsolved]) > 0.0)



if __name__ == '__main__':
    unittest.main()
