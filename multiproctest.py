import multiprocessing as mp
from FlowCalculation import FlowCalculation
from solve_flows import solve_flow

modes = ['DC_sqr', 'sqr']
fclist = []
for mode in modes:
    fclist.append(FlowCalculation('Europe', 'aHE', '0.7q99', mode))

pool = mp.Pool(mp.cpu_count())
print [str(fc) for fc in fclist]

pool.map(solve_flow, fclist)
