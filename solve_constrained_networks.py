import numpy as np
import multiprocessing as mp
from solve_flows import solve_flow
from FlowCalculation import FlowCalculation

modes = ['lin', 'sqr', 'DC_lin', 'DC_sqr']
capacities = [''.join([str(a), 'q99']) for a in np.linspace(0,1.5,31)]

fc_list = []
for m in modes:
    for c in capacities:
        fc = FlowCalculation('Europe', 'aHE', c, m, savemode='FCResult flows')
        fc_list.append(fc)

print [str(fc) for fc in fc_list]

pool = mp.Pool(mp.cpu_count())
pool.map(solve_flow, fc_list)
