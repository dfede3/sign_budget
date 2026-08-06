import numpy as np
import gurobipy as gp
from gurobipy import GRB
import problem_functions
import time

class RMCP_functions(problem_functions.Pb_functions):
    def __init__(self, tmax):
        self.tmax = tmax

    def MCP(self, env, gp_env, p):
        model = gp.Model("Maximum Coverage Problem", env = gp_env)
        # model parameters:
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self.tmax

        # decision variables
        x = model.addVars(env.I, vtype=GRB.BINARY, name = "x") # x[i] = 1 if facility i is opened
        z = model.addVars(env.J, vtype=GRB.BINARY, name = "z") # z[j] = 1 if demand point j is covered

        covered = {j for j in range(env.J) if len(env.N[j]) > 0}

        # objective function
        obj_expr = gp.quicksum((env.dem[j]-env.dev[j]) * z[j] for j in covered)
        model.setObjective(obj_expr, GRB.MAXIMIZE)
        
        # constraints
        model.addConstr(gp.quicksum(x[i] for i in range(env.I))<= p)
        # constraints using N instead of a
        model.addConstrs(gp.quicksum(x[i] for i in env.N[j]) >= z[j] for j in covered)

        model.update()
        t1 = time.perf_counter()   # start solve
        model.optimize()
        t2 = time.perf_counter()   # end solve
        solve_time = t2 - t1

        if not(model.Status == gp.GRB.OPTIMAL):
            if model.Status == gp.GRB.TIME_LIMIT :
                status = 'time_limit'
            elif model.Status == gp.GRB.INFEASIBLE :
                status = 'infeasible'
            else :
                print(f"Model status is {model.Status}")
                status = None

            if model.SolCount > 0:
                obj_val = model.ObjVal
                x_sol = np.array([i for i, var in x.items() if var.X > 0.5], dtype=int)
                z_sol = np.array([i for i, var in z.items() if var.X > 0.5], dtype=int)
            else:
                print("No feasible solution found.")
                obj_val = None
                x_sol = None
                z_sol = None
        else :
            status = 'optimal'
            obj_val = model.ObjVal
            x_sol = np.array([i for i, var in x.items() if var.X > 0.5], dtype=int)
            z_sol = np.array([i for i, var in z.items() if var.X > 0.5], dtype=int)
                
        return status, obj_val, x_sol, z_sol, solve_time


    def cRC(self, env, gp_env, p, gamma = 0.1):
        model = gp.Model("Classical Robust Counterpart of Maximum Coverage Problem", env = gp_env)
        # model parameters:
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self.tmax

        # decision variables
        x = model.addVars(env.I, vtype=GRB.BINARY, name = "x") # x[i] = 1 if facility i is opened
        z = model.addVars(env.J, vtype=GRB.BINARY, name = "z") # z[j] = 1 if demand point j is covered
        y = model.addVars(env.J, vtype=GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name = "y") 
        u = model.addVars(env.J, vtype=GRB.CONTINUOUS, name = "u")
        t = model.addVar(vtype=GRB.CONTINUOUS, name = "t")

        covered = {j for j in range(env.J) if len(env.N[j]) > 0}

        # objective function
        obj_expr = gp.quicksum(env.dem[j]*z[j] for j in covered) - gp.quicksum(u[j] for j in covered) - (int(gamma * env.Dem) +env.Dev)*t
        model.setObjective(obj_expr, GRB.MAXIMIZE)
        
        # constraints
        model.addConstr(gp.quicksum(x[i] for i in range(env.I))<= p)
        model.addConstrs(gp.quicksum(x[i] for i in env.N[j]) >= z[j] for j in covered)

        # extra constraints for cRC
        model.addConstrs(u[j] >= y[j] for j in covered)
        model.addConstrs(u[j] >= -y[j] for j in covered)
        model.addConstrs(t >= env.dev[j]*z[j] + y[j] for j in covered)
        model.addConstrs(t >= - (env.dev[j]*z[j] + y[j]) for j in covered)

        model.update()
        t1 = time.perf_counter()   # start solve
        model.optimize()
        t2 = time.perf_counter()   # end solve
        solve_time = t2 - t1

        if not(model.Status == gp.GRB.OPTIMAL):
            if model.Status == gp.GRB.TIME_LIMIT :
                status = 'time_limit'
            elif model.Status == gp.GRB.INFEASIBLE :
                status = 'infeasible'
            else :
                print(f"Model status is {model.Status}")
                status = None

            if model.SolCount > 0:
                obj_val = model.ObjVal
                x_sol = np.array([i for i, var in x.items() if var.X > 0.5], dtype=int)
                z_sol = {j: int(z[j].X > 0.5) for j in range(env.J)}
            else:
                print("No feasible solution found.")
                obj_val = None
                x_sol = None
                z_sol = None
        else :
            status = 'optimal'
            obj_val = model.ObjVal
            x_sol = np.array([i for i, var in x.items() if var.X > 0.5], dtype=int)
            z_sol = {j : z[j].X for j in range(env.J)}
                
        return status, obj_val, x_sol, z_sol, solve_time

    def compute_obj_SB(self, env, z_sol):
        covered = {j for j in range(env.J) if len(env.N[j]) > 0}
        u = max(0,(env.DemSB - env.DevSB) - sum((env.dem[j] + env.dev[j]) * (1-z_sol[j]) for j in covered) -  sum((env.dem[j]-env.dev[j]) * z_sol[j] for j in covered))
        obj_val = sum((env.dem[j] - env.dev[j]) * z_sol[j] for j in covered) + u
        return obj_val

    def RC2Gurobi(self, env, gp_env, p):
        model = gp.Model("Robust Counterpart 2 of Maximum Coverage Problem", env = gp_env)
        # model parameters:
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self.tmax

        # decision variables
        x = model.addVars(env.I, vtype=GRB.BINARY, name = "x")
        z = model.addVars(env.J, vtype=GRB.BINARY, name = "z")

        covered = {j for j in range(env.J) if len(env.N[j]) > 0}

        # precompute constants once
        D = env.DemSB - env.DevSB
        sum_a = sum(env.dem[j] + env.dev[j] for j in covered)   
        sum_dev = sum(env.dev[j] for j in covered)               

        v_lb = D - sum_a
        v_ub = v_lb + 2 * sum_dev
        u_ub = max(0.0, v_ub)

        v = model.addVar(lb=v_lb, ub=v_ub, vtype=GRB.CONTINUOUS, name="v")
        u = model.addVar(lb=0, ub=u_ub, vtype=GRB.CONTINUOUS, name="u")

        obj_expr = gp.quicksum((env.dem[j] - env.dev[j]) * z[j] for j in covered) + u
        model.setObjective(obj_expr, GRB.MAXIMIZE)

        model.addConstr(gp.quicksum(x[i] for i in range(env.I)) <= p)
        model.addConstrs(gp.quicksum(x[i] for i in env.N[j]) >= z[j] for j in covered)

        # simplified, sparser RC constraint
        model.addConstr(v == (v_lb) + 2 * gp.quicksum(env.dev[j] * z[j] for j in covered))
        model.addConstr(u == gp.max_(v, 0),name = "u_min_v_0")

        model.update()
        t1 = time.perf_counter()   # start solve
        model.optimize()
        t2 = time.perf_counter()   # end solve
        solve_time = t2 - t1

        if not(model.Status == gp.GRB.OPTIMAL):
            if model.Status == gp.GRB.TIME_LIMIT :
                status = 'time_limit'
            elif model.Status == gp.GRB.INFEASIBLE :
                status = 'infeasible'
            else :
                print(f"Model status is {model.Status}")
                status = None

            if model.SolCount > 0:
                obj_val = model.ObjVal
                x_sol = np.array([i for i, var in x.items() if var.X > 0.5], dtype=int)
                z_sol = np.array([i for i, var in z.items() if var.X > 0.5], dtype=int)
            else:
                print("No feasible solution found.")
                obj_val = None
                x_sol = None
                z_sol = None
        else :
            status = 'optimal'
            obj_val = model.ObjVal
            x_sol = np.array([i for i, var in x.items() if var.X > 0.5], dtype=int)
            z_sol = np.array([i for i, var in z.items() if var.X > 0.5], dtype=int)
                
        return status, obj_val, x_sol, z_sol, solve_time
    


