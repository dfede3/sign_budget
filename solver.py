from datetime import datetime
import gurobipy as gp
import numpy as np
import copy

import time
from tabulate import tabulate
import psutil

class Solver:
    def __init__(self, problem, method = 'nominal', env = None, ram = 95,
                 max_time = 60*60, p = 1, print_info = False):
        self.env = env
        self.problem = problem 
        self.env = env
        self.ram = ram
        self.max_time   = max_time
        self.p = p
        self.print_info = print_info
        self.env.read_test_inst()
    
    def set_sol_method(self, method):
        self.method = method
    
    def solve(self, method = None): 
        if method is not None : 
            self.set_sol_method(method)

        if self.method   == 'nominal':
            info, ram_issue = self.solveMCP()
        elif self.method == 'cRC':
            info, ram_issue = self.solveRCMCP(version = 0)
        elif self.method == 'RC1':
            info, ram_issue = self.solveRCMCP(version = 1)
        elif self.method == 'RC2':
            info, ram_issue = self.solveRCMCP(version = 2)
        elif self.method == 'RC2Gurobi':
            info, ram_issue = self.solveRCMCP(version = 3)
        else :
            raise NotImplementedError('FORMULATION UNKNOWN')

        return info, ram_issue
        
    def test_problem(self, solvers) : 
        res_tab  = []

        for solver in solvers :
            info, ram_issue = self.solve(method = solver)
            print(f'Formulation {solver} ended!')

            if ram_issue :
                print('Warning: Ram Issue!')
                return
            
            res_tab.append([solver, info['time_solve'], np.round(info['f_opt'],3)])

        table = tabulate(res_tab, headers = ['Algorithm', 't_solve', 'f_opt'], tablefmt='orgtbl')
        print(table) 
        
        return 
    
    def solveMCP(self) :
        env = copy.deepcopy(self.env) 
        gp_env = gp.Env()
        gp_env.setParam("OutputFlag", 0)
        gp_env.setParam("Threads", 1)

        now = datetime.now().time()
        print(f"Instance {env.inst_num}: MCP started at {now}")
        start_time = time.time()
        status, fopt, x, z, solve_time = self.problem.MCP(env, gp_env, self.p)
        runtime = time.time() - start_time

        if status == 'time_limit' :
            print("MCP could not be solved before TL")
        elif status == 'infeasible' :
            print("MCP is not feasible")
        if self.print_info:
            now = datetime.now().time()
            now_nice = f"{now.hour}:{now.minute}:{now.second}"
            print(f"MCP, completed at {now_nice}, solved in {np.round(runtime/60, 3)} minutes")

        return {"time_solve": solve_time, "f_opt": fopt, "x_opt": x, "z_opt": z}, psutil.virtual_memory().percent > self.ram

    def solveRCMCP(self, version = 1) :
        env = copy.deepcopy(self.env) 
        gp_env = gp.Env()
        gp_env.setParam("OutputFlag", 0)
        gp_env.setParam("Threads", 1)

        now = datetime.now().time()
        print(f"Instance {env.inst_num}: RC{version}-MCP started at {now}")
        start_time = time.time()
        if version == 1 :
            status, fopt, x, z, solve_time = self.problem.RC1(env, gp_env, self.p)
        elif version == 2:
            status, fopt, x, z, solve_time = self.problem.RC2(env, gp_env, self.p)
        elif version == 3:
            status, fopt, x, z, solve_time = self.problem.RC2Gurobi(env, gp_env, self.p)
        elif version == 0:
            status, fopt, x, z, solve_time = self.problem.cRC(env, gp_env, self.p)
        runtime = time.time() - start_time

        if status == 'time_limit' :
            print(f"RC{version}-MCP could not be solved before TL")
        elif status == 'infeasible' :
            print(f"RC{version}-MCP is not feasible")
        if self.print_info:
            now = datetime.now().time()
            now_nice = f"{now.hour}:{now.minute}:{now.second}"
            print(f"RC{version}-MCP, completed at {now_nice}, solved in {np.round(runtime/60, 3)} minutes")

        return {"time_solve": solve_time, "f_opt": fopt, "x_opt": x, "z_opt": z}, psutil.virtual_memory().percent > self.ram
    