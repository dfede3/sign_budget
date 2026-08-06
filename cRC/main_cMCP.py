from environment import cRMCP_env as Env
import pf_MCP
from solver import Solver
from datetime import datetime

# list of formulations we compare: options are
# - 'nominal': deterministic problem
# - 'RC2Gurobi' : compact robust counterpart under signed budget uncertainty
# - 'cRC_gamma' : RC of the classical budget (Gamma) uncertainty
#                 "gamma" is s.t. Gamma = I * gamma/100

formulations = ['RC2Gurobi', 'cRC_10', 'cRC_20', 'cRC_30','cRC_40', 'cRC_50', 'cRC_60', 'cRC_70', 'cRC_80', 'cRC_90', 'cRC_100'] 

tmax = 60*60 # time limit 

# only one instance for each combination of parameters

r = 0.2 # coverage radius
p = 5   # maximum number of bases that can be opened

# create instances :
for I in [100, 250, 500, 750, 1000]: # number of bases/demand points
    now = datetime.now().time()
    print(f'Environments creation started at {now}')
    env = Env(I=I, r=r, inst_num = 1) 
    env.make_test_inst()
    now = datetime.now().time()
    now_nice = f"{now.hour}:{now.minute}:{now.second}"
    print(f'Environments creation completed at {now_nice}')

    print(f'N_locations = {I}, r = {r}, p = {p}')

    pp = pf_MCP.RMCP_functions(tmax=tmax)

    S = Solver(problem = pp, env = env, max_time = tmax, p=p)
    S.test_problem(formulations) 