from europe_plusgrid import europe_plus_Nodes
import DCsolvers as DCs
import numpy as np

N = europe_plus_Nodes()

copperNodes, copperFlow = DCs.DC_solve(N, mode='linear copper verbose', msg="Solving copper flows")
copperNodes.save_nodes("copperNodes.npz")
np.save("./results/copper_flows.npy", copperFlow)
constrNodes, constrFlow = DCs.DC_solve(N, mode='linear verbose', msg="Solving constrained flows", b=1e3)
constrNodes.save_nodes("constrNodes.npz")
np.save("./results/constr_flows.npy", constrFlow)
