from joblib import Parallel, delayed
from env_kp import KP_env as Env
import pf_knapsack
from solver_kp import Solver
from datetime import datetime

# list of formulations we compare: options are '2K_nominal', 'RC1', 'RC2Gurobi'
formulations = ['2K_nominal','RC2Gurobi']

tmax = 60*60 # time limit

num_instances = 5 # number of tested instances

N = 192 # number of items 
alpha = 0.5 # parameter to define the capacity of the knapsack

#for K in [[2], [3], [4], [6], [8], [12], [16], [24], [32], [48]]: # number of subclasses
for K in [[2], [3], [4], [6], [8], [12], [16]]: # number of subclasses
    now = datetime.now().time()
    print(f'Environments creation started at {now}')
    env_list = [Env(N=N, alpha=alpha, K=K, inst_num = i) for i in range(1,num_instances+1)]
    Parallel(n_jobs=-1)(delayed(env.read_test_inst)() for env in env_list)
    now = datetime.now().time()
    now_nice = f"{now.hour}:{now.minute}:{now.second}"
    print(f'Environments creation completed at {now_nice}')
    
    pp = pf_knapsack.KP_functions(tmax=tmax)
    print(f'N_items = {N}, K = {K[0]}')

    for env in env_list :
        S = Solver(problem = pp, env = env, max_time = tmax)
        S.test_problem(formulations)

