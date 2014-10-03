from europe_plusgrid import europe_plus_Nodes
import DCsolvers as DCs
import numpy as np

N = europe_plus_Nodes()

h0 = np.load('./results/EuropeCopperh0.npy')
#copperNodes, copperFlow = DCs.DC_solve(N, mode='linear copper verbose', msg="Solving copper flows")
#copperNodes.save_nodes("copperNodes.npz")
#np.save("./results/copper_flows.npy", copperFlow)

constrNodes_lin, constrFlow_lin = DCs.DC_solve(N, mode='linear verbose', msg="Solving constrained flows lin", h0=h0, b=0.05)
constrNodes_lin.save_nodes("Europe_aHE_0.05q99_lin.npz")
np.save("./results/Europe_aHE_0.5q99_lin_flows.npy", constrFlow_lin)

copperNodes_sqr, copperFlow_sqr = DCs.DC_solve(N, mode='square copper verbose', msg="Solving copper flows square")
copperNodes_sqr.save_nodes("Europe_aHE_copper_sqr.npz")
np.save("./results/Europe_aHE_copper_sqr_flows.npy", copperFlow_sqr)



