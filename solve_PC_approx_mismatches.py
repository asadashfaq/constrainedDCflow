import numpy as np
import multiprocessing as mp
from solve_flows import solve_flow
from FlowCalculation import FlowCalculation

Ks = ['K' + str(K) for K in range(1,8)]
basepath = 'results/Europe_gHO1.0_aHO0.8_approx_mismatch_'
mm_paths = [basepath + K + '.npy' for K in Ks]

fc_list = [FlowCalculation('Europe', 'aHO0.8', 'copper', 'lin',\
          savemode='full', mismatch_path=p) for p in mm_paths]
print [str(fc) for fc in fc_list]
print len(fc_list)

pool = mp.Pool(mp.cpu_count())
pool.map(solve_flow, fc_list)
