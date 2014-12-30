import sys
import numpy as np
import multiprocessing as mp

import aurespf.solvers as au
import DCsolvers as DCs
from europe_plusgrid import europe_plus_Nodes
from FlowCalculation import FlowCalculation # my own class for passing info about calculations
from FCResult import FCResult # my class for storing results in a space-efficient way

def solve_flow(flow_calc):
    admat = ''.join(['./settings/', flow_calc.layout, 'admat.txt'])
    filename = str(flow_calc)

    copperflow_filename = ''.join(['./results/', flow_calc.layout, '_',
        flow_calc.alphas, '_copper_', flow_calc.solvermode, '_flows.npy'])

    if flow_calc.alphas=='aHE':
        if flow_calc.basisnetwork == 'europeplus':
            nodes = europe_plus_Nodes(admat=admat)
        else:
            sys.stderr.write('The object has a basisnetwork that\
                          is not accounted for. Use "europeplus".')
    elif flow_calc.alphas.startswith('aHO'):
        alpha = float(flow_calc.alphas[3:]) # expects alphas on the form aHO0.4
        if flow_calc.basisnetwork == 'europeplus':
            nodes = europe_plus_Nodes(admat=admat, alphas=alpha)
            if flow_calc.mismatch_path != None:
                nodes = set_mismatches(nodes, flow_calc.mismatch_path)

        else:
            sys.stderr.write('The object has a basisnetwork that\
                          is not accounted for. Use "europeplus".')
    else:
        sys.stderr.write('The object has an distribution of mixes that\
                          is not accounted for.')

    mode_str_list = []
    if 'lin' in flow_calc.solvermode:
        mode_str_list.append('linear ')
    elif 'sqr' in flow_calc.solvermode:
        mode_str_list.append('square ')
    else:
        sys.stderr.write('The solver mode must be "lin", "sqr"')
    if 'imp' in flow_calc.solvermode:
        mode_str_list.append('impedance ')

    mode = ''.join(mode_str_list)

    flowfilename = ''.join(['./results/', filename, '_flows.npy'])

    if 'DC' in flow_calc.solvermode:
        solver = DCs.DC_solve
    else:
        solver = au.solve

    if flow_calc.capacities=='copper':
        solved_nodes, flows = solver(nodes, mode=''.join([mode, ' copper']),\
                                msg=str(flow_calc))
    elif flow_calc.capacities=='q99':
        h0 = au.get_quant_caps(filename=copperflow_filename)
        solved_nodes, flows = solver(nodes, h0=h0, mode=mode, \
                                         msg=str(flow_calc))
    elif flow_calc.capacities=='hq99': # corresponds to half the capacities
                                         # of the 99% quantile layout
        h0 = 0.5*au.get_quant_caps(filename=copperflow_filename)
        solved_nodes, flows = solver(nodes, h0=h0, mode=mode, \
                                        msg=str(flow_calc))
    elif flow_calc.capacities.endswith('q99'):
        scale = float(flow_calc.capacities[0:-3])
        h0 = au.get_quant_caps(filename=copperflow_filename)
        solved_nodes, flows = solver(nodes, h0=h0, b=scale, mode=mode,\
                                        msg=str(flow_calc))
    else:
        sys.stderr.write('The capacities must be either "copper", "q99",\
                            "hq99", or on the form "<number>q99"')

    if flow_calc.savemode == 'full':
        solved_nodes.save_nodes(filename)
        try:
            flows
            np.save('./results/' + filename + '_flows', flows)
        except NameError:
            print "Flows not defined."

    elif 'FCResult' in flow_calc.savemode:
        result = FCResult(filename+'.pkl')
        result.add_instance(solved_nodes, flows, flow_calc)
        result.save_results(filename+'.pkl')
        if 'flows' in flow_calc.savemode:
            try:
                flows
                np.save('./results/' + filename + '_flows', flows)
            except NameError:
                print "Flows not defined."

    else:
        print "Results not saved, invalid savemode provided"


def set_mismatches(nodes, mismatch_path):
    new_mismatch = np.load(mismatch_path)
    for i in xrange(len(nodes)):
        nodes[i].mismatch = new_mismatch[i]

    return nodes

