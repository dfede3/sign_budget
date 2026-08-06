import numpy as np
import gurobipy as gp
from gurobipy import GRB
import problem_functions
import time

class RMCP_L2_functions(problem_functions.Pb_functions):
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
        obj_expr = gp.quicksum(env.dem[j] * z[j] for j in covered)
        model.setObjective(obj_expr, GRB.MAXIMIZE)
        
        # max coverage constraints
        model.addConstr(gp.quicksum(x[i] for i in range(env.I))<= p)
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
    
    def RC1(self, env, gp_env, p):
        model = gp.Model("Robust Counterpart 1 L=2 of Maximum Coverage Problem", env = gp_env)
        # model parameters:
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self.tmax

        unc = env.unc  # two level UncertaintySet_L2 (one set in the first level and 4 sets in the first one)

        # decision variables
        x = model.addVars(env.I, vtype=GRB.BINARY, name = "x")
        z = model.addVars(env.J, vtype=GRB.BINARY, name = "z")

        # additional variabels for the RC
        u1 = model.addVars(4,
                          vtype=GRB.CONTINUOUS, lb = -float('inf'), ub = 0, name = "u1")
        w1 = model.addVars(4,
                         vtype=GRB.CONTINUOUS, lb = -float('inf'), ub = 0, name = "w1")
        u2 = model.addVar(vtype=GRB.CONTINUOUS, lb = -float('inf'), ub = 0, name = "u2")
        w2 = model.addVar(vtype=GRB.CONTINUOUS, lb = -float('inf'), ub = 0, name = "w2")
        vp = model.addVars(env.J, vtype=GRB.CONTINUOUS, lb = -float('inf'), ub = 0, name = "vp")
        vm = model.addVars(env.J, vtype=GRB.CONTINUOUS, lb = -float('inf'), ub = 0, name = "vm")

        covered = {j for j in range(env.J) if len(env.N[j]) > 0}

        # (robust) objective function 
        obj_expr = gp.quicksum(
             (env.Dem[1][k] + env.Dev[1][k]) * u1[k-1]
             - (env.Dem[1][k] - env.Dev[1][k]) * w1[k-1]
         for k in range(1, 5)) + (env.Dem[2][1] + env.Dev[2][1]) * u2 - (env.Dem[2][1] - env.Dev[2][1]) * w2 + gp.quicksum(
            (env.dem[j] + env.dev[j]) * vp[j]
            - (env.dem[j] - env.dev[j]) * vm[j]
        for j in covered)

        model.setObjective(obj_expr, GRB.MAXIMIZE)

        # max coverage constraints
        model.addConstr(gp.quicksum(x[i] for i in range(env.I))<= p)
        model.addConstrs(gp.quicksum(x[i] for i in env.N[j]) >= z[j] for j in covered)
        
        # RC constraints 
        model.addConstrs(
            gp.quicksum(
                u1[k-1] - w1[k-1]
                for (l, k) in unc.find_groups(j) if l == 1
            )
            + (u2 - w2) 
            + vp[j] - vm[j] == z[j]
            for j in covered
        )

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
    
    def RC2(self, env, gp_env, p):
        model = gp.Model("Robust Counterpart 2 L=2 of Maximum Coverage Problem", env = gp_env)
        # model parameters:
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self.tmax

        unc = env.unc

        # decision variables
        x = model.addVars(env.I, vtype=GRB.BINARY, name = "x")
        z = model.addVars(env.J, vtype=GRB.BINARY, name = "z")

        # additional variabels for the RC
        # level 1: k=1,2,3,4
        w1 = model.addVars(4, vtype=GRB.CONTINUOUS, lb=0, name="w1")
        v1 = model.addVars(4, vtype=GRB.BINARY, name="v1")
        u1 = model.addVars(4, vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=0, name="u1")
        vv1 = model.addVars(4, vtype=GRB.BINARY, name="vv1")

        # level 2: K_2=1
        w2  = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="w2")
        v2 = model.addVar(vtype=GRB.BINARY, name="v2")

        covered = {j for j in range(env.J) if len(env.N[j]) > 0}


        # (robust) objective function
        obj_expr = gp.quicksum((env.dem[j] - env.dev[j]) * z[j] for j in covered) + gp.quicksum(w1[k] for k in range(4)) + w2

        model.setObjective(obj_expr, GRB.MAXIMIZE)

        # max coverage constraints
        model.addConstr(gp.quicksum(x[i] for i in range(env.I))<= p)
        model.addConstrs(gp.quicksum(x[i] for i in env.N[j]) >= z[j] for j in covered)

        # RC constraints : first level 
        for k in range(4):
            grp_cov = [j for j in unc.get_group(1, k+1) if j in covered]
            M = 2*sum(env.dev[j] for j in grp_cov) # big-M coeff
            
            model.addConstr(
                u1[k] >= (env.Dem[1][k+1] + env.Dev[1][k+1])
                - gp.quicksum((env.dem[j] + env.dev[j]) * (1 - z[j]) for j in grp_cov) 
                - gp.quicksum((env.dem[j] - env.dev[j]) * z[j] for j in grp_cov)
                - M * v1[k]
            )
            model.addConstr(u1[k] >= - M * (1-v1[k]))

            model.addConstr(
                w1[k] <= ((env.Dem[1][k+1] - env.Dev[1][k+1]))
                - gp.quicksum((env.dem[j] + env.dev[j]) * (1 - z[j]) for j in grp_cov) 
                - gp.quicksum((env.dem[j] - env.dev[j]) * z[j] for j in grp_cov) 
                + M * vv1[k]
            )
            model.addConstr(w1[k] <= M * (1 - vv1[k]))

        # RC constraints : second level
        M = 2*(sum(env.dev))
        model.addConstr(
            w2 <= ((env.Dem[2][1] - env.Dev[2][1]))
            - (gp.quicksum((env.dem[j] + env.dev[j]) * (1-z[j]) for j in covered) + gp.quicksum(u1[k] for k in range(4)))
            - (gp.quicksum((env.dem[j] - env.dev[j]) * z[j] for j in covered) + gp.quicksum(w1[k] for k in range(4)))
            + M * v2
        )

        model.addConstr(
           w2 <=  M * (1 - v2)
        )

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
    
    def RC2Gurobi(self, env, gp_env, p):
        model = gp.Model("Robust Counterpart 2 L=2 of Maximum Coverage Problem", env=gp_env)
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self.tmax

        unc = env.unc

        x = model.addVars(env.I, vtype=GRB.BINARY, name="x")
        z = model.addVars(env.J, vtype=GRB.BINARY, name="z")

        covered = {j for j in range(env.J) if len(env.N[j]) > 0}

        # level 1: k = 0..3
        w1 = model.addVars(4, lb=0, vtype=GRB.CONTINUOUS, name="w1")                     # w1[k] = max(0, e1[k])
        u1 = model.addVars(4, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, name="u1")   # u1[k] = min(0, e2[k])
        e1 = model.addVars(4, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="e1")
        e2 = model.addVars(4, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="e2")

        # level 2: single aggregate group
        w2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="w2")                          # w2 = max(0, e3)
        e3 = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="e3")

        # (robust) objective function
        obj_expr = (gp.quicksum((env.dem[j] - env.dev[j]) * z[j] for j in covered)
                    + gp.quicksum(w1[k] for k in range(4)) + w2)
        model.setObjective(obj_expr, GRB.MAXIMIZE)

        model.addConstr(gp.quicksum(x[i] for i in range(env.I)) <= p)
        model.addConstrs(gp.quicksum(x[i] for i in env.N[j]) >= z[j] for j in covered)

        for k in range(4):
            grp_cov = [j for j in unc.get_group(1, k + 1) if j in covered]

            model.addConstr(
                e1[k] == (env.Dem[1][k + 1] - env.Dev[1][k + 1])
                - gp.quicksum((env.dem[j] + env.dev[j]) * (1 - z[j]) for j in grp_cov)
                - gp.quicksum((env.dem[j] - env.dev[j]) * z[j] for j in grp_cov)
            )
            model.addConstr(w1[k] == gp.max_(e1[k], 0), name=f"w1_def_{k}")

            # e2[k]: 
            model.addConstr(
                e2[k] == (env.Dem[1][k + 1] + env.Dev[1][k + 1])
                - gp.quicksum((env.dem[j] + env.dev[j]) * (1 - z[j]) for j in grp_cov)
                - gp.quicksum((env.dem[j] - env.dev[j]) * z[j] for j in grp_cov)
            )
            model.addConstr(u1[k] == gp.min_(e2[k], 0), name=f"u1_def_{k}")

        # e3: 
        model.addConstr(
            e3 == (env.Dem[2][1] - env.Dev[2][1])
            - (gp.quicksum((env.dem[j] + env.dev[j]) * (1 - z[j]) for j in covered) + gp.quicksum(u1[k] for k in range(4)))
            - (gp.quicksum((env.dem[j] - env.dev[j]) * z[j] for j in covered) + gp.quicksum(w1[k] for k in range(4)))
        )
        model.addConstr(w2 == gp.max_(e3, 0), name="w2_def")

        model.update()
        t1 = time.perf_counter()
        model.optimize()
        t2 = time.perf_counter()
        solve_time = t2 - t1

        if not (model.Status == gp.GRB.OPTIMAL):
            if model.Status == gp.GRB.TIME_LIMIT:
                status = 'time_limit'
            elif model.Status == gp.GRB.INFEASIBLE:
                status = 'infeasible'
            else:
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
        else:
            status = 'optimal'
            obj_val = model.ObjVal
            x_sol = np.array([i for i, var in x.items() if var.X > 0.5], dtype=int)
            z_sol = np.array([i for i, var in z.items() if var.X > 0.5], dtype=int)

        return status, obj_val, x_sol, z_sol, solve_time
        