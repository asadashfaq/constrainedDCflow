import unittest
import numpy as np
from europe_plusgrid import europe_plus_Nodes
import aurespf.solvers as au
import DCsolvers as DCs

class Test_DC_solve(unittest.TestCase):

    def setUp(self):
        self.N = europe_plus_Nodes()
        self.testlapse = [60192, 136380, 155686, 230094, 243333, 244000, 10] #range(60150, 60210)
        self.h0 = np.load('./results/EuropeCopperh0.npy')

    def test_constrained_flow(self):
        Nsolved, F = DCs.DC_solve(self.N, mode='linear',\
                            msg='Solving constrained flows', b=1e3,
                            lapse=self.testlapse)
        for t in self.testlapse:
            self.assertTrue(sum([n.balancing[t] + n.curtailment[t]\
                                 for n in Nsolved]) > 0.0)

    def test_constrained_h0_flow(self):
        Nsolved, F = DCs.DC_solve(self.N, mode='linear',\
                            h0=self.h0, b=0.9,
                            msg='Solving constrained flows',
                            lapse=self.testlapse)

        for t in self.testlapse:
            self.assertTrue(sum([n.balancing[t] + n.curtailment[t]\
                                 for n in Nsolved]) > 0.0)

    def test_constrained_h0_sqr_flow(self):
        Nsolved, F = DCs.DC_solve(self.N, mode='square',\
                            h0=self.h0, b=0.05,
                            msg='Solving constrained flows',
                            lapse=self.testlapse)

        for t in self.testlapse:
            self.assertTrue(sum([n.balancing[t] + n.curtailment[t]\
                                 for n in Nsolved]) > 0.0)



if __name__ == '__main__':
    unittest.main()
