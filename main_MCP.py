from joblib import Parallel, delayed
from environment import RMCP_env as Env
import pf_MCP
from solver import Solver
from datetime import datetime

# list of formulations we compare. options are
# - 'nominal': deterministic problem
# - 'RC1': robust counterpart derived from duality
# - 'RC2': compact RC (big-M formulation)
# - 'RC2Gurobi': compact RC (uses built in Gurobi's functions)

formulations = ['nominal','RC1','RC2', 'RC2Gurobi'] 

tmax = 60*60 # time limit

num_instances = 5 # number of tested instances for each combination of the parameters

# create instances :
for r in [0.1, 0.2]: # coverage radius
    for I in [100, 500, 1000, 5000, 10000]: # number of bases/demand points
        now = datetime.now().time()
        print(f'Environments creation started at {now}')
        env_list = [Env(I=I,r=r,inst_num = i) for i in range(1,num_instances+1)]
        Parallel(n_jobs=-1)(delayed(env.make_test_inst)() for env in env_list)
        now = datetime.now().time()
        now_nice = f"{now.hour}:{now.minute}:{now.second}"
        print(f'Environments creation completed at {now_nice}')
        for p in [5, 10]: # number of bases to open
            print(f'N_locations = {I}, radius = {r}, max num of facilities = {p}')

            pp = pf_MCP.RMCP_functions(tmax=tmax)

            for env in env_list :
                S = Solver(problem = pp, env = env, max_time = tmax, p=p)
                S.test_problem(formulations) 