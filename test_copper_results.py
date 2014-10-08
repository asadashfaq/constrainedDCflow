from europe_plusgrid import europe_plus_Nodes
from FlowCalculation import FlowCalculation


for flow in ['lin', 'sqr']:
    DC_filename = 'Europe_aHE_copper_DC_' + flow + '.npz'
    F2_filename = 'Europe_aHE_copper_' + flow + '.npz'
    DC_N = europe_plus_Nodes(load_filename=DC_filename)
    F2_N = europe_plus_Nodes(load_filename=F2_filename)


