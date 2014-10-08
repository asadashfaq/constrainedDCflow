import cProfile
import solve_flows

fc = solve_flows.FlowCalculation('Europe', 'aHE', '0.6q99', 'DC_sqr')
cProfile.run('solve_flows.solve_flow(fc)', 'profile1')
