import numpy as np
import gurobipy as gp
from gurobipy import GRB
import problem_functions
import time
import itertools

class KP_functions(problem_functions.Pb_functions):
    def __init__(self, tmax):
        self.tmax = tmax


    def NP_2K(self, env, gp_env):
        """
        Enumerates all 2^K nominal problem.
        Model is built ONCE and updated in place (RHS + coefficients) per scenario,
        instead of rebuilt from scratch every iteration.
        """
        t0 = time.perf_counter()
        N, K1 = env.N, env.K1

        # Precompute everything that does NOT depend on w
        group_of_item = np.empty(N, dtype=int)
        weight_off = np.empty(N)   # weight if w[k] == 0  -> avg + dev
        weight_on = np.empty(N)    # weight if w[k] == 1  -> avg - dev
        group_cost = np.empty(K1)  # capacity reduction if w[k] == 1

        for k in range(K1):
            group_cost[k] = (env.avg_weight_1k[k] + env.dev_weight_1k[k]) - sum(
                env.avg_weight[j] - env.dev_weight[j] for j in env.group1[k]
            )
            for j in env.group1[k]:
                group_of_item[j] = k
                weight_off[j] = env.avg_weight[j] + env.dev_weight[j]
                weight_on[j] = env.avg_weight[j] - env.dev_weight[j]

        # Build the Gurobi model once
        model = gp.Model("NP_2K Knapsack", env=gp_env)
        model.Params.OutputFlag = 0

        # variables
        x = model.addVars(N, vtype=GRB.BINARY, name="x")

        # objective function
        model.setObjective(gp.quicksum(env.value[j] * x[j] for j in range(N)), GRB.MAXIMIZE)
        # constraint
        cap_constr = model.addConstr(
            gp.quicksum(weight_off[j] * x[j] for j in range(N)) <= env.capacity, "capacity"
        )
        model.update()

        opt = -np.inf
        local_best_value, local_chosen_items = None, None
        found_optimal = False

        for w in itertools.product([0, 1], repeat=K1):
            elapsed = time.perf_counter() - t0
            remaining = self.tmax - elapsed
            if remaining <= 0:
                return 'time_limit', np.inf, None, None, elapsed

            w_arr = np.asarray(w)
            capacity = env.capacity - group_cost[w_arr == 1].sum()
            weight = np.where(w_arr[group_of_item] == 0, weight_off, weight_on)

            # Update the existing model instead of rebuilding it
            cap_constr.RHS = capacity
            for j in range(N):
                model.chgCoeff(cap_constr, x[j], weight[j])
            model.Params.TimeLimit = remaining

            model.optimize()

            if model.status == GRB.OPTIMAL:
                if model.objVal > opt:
                    opt = model.objVal
                    local_best_value = model.objVal
                    local_chosen_items = [j for j in range(N) if x[j].X > 0.5]
                    found_optimal = True

        solve_time = time.perf_counter() - t0
        if not found_optimal:
            return 'infeasible', np.inf, None, None, solve_time
        return 'optimal', local_best_value, local_chosen_items, None, solve_time

    def RC1(self, env, gp_env):
        model = gp.Model("Classical Robust Counterpart of the Knapsack Problem", env = gp_env)
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self.tmax 

        n = env.N
        x = model.addVars(n, vtype=GRB.BINARY, name="x")

        w1 = model.addVars(env.K1, vtype=GRB.CONTINUOUS, name = "w1")
        u1 = model.addVars(env.K1, vtype=GRB.CONTINUOUS, name="u1")
        vp = model.addVars(n, vtype=GRB.CONTINUOUS, name = "vp")
        vm = model.addVars(n, vtype=GRB.CONTINUOUS, name = "vm")

        # objective function
        model.setObjective(gp.quicksum(env.value[i] * x[i] for i in range(n)), GRB.MAXIMIZE)

        # (robust) constraint
        model.addConstr(gp.quicksum((env.avg_weight_1k[k] + env.dev_weight_1k[k]) * w1[k]
                                    - (env.avg_weight_1k[k] - env.dev_weight_1k[k]) * u1[k]
                                    for k in range(env.K1)) +
            gp.quicksum((env.avg_weight[j] + env.dev_weight[j]) * vp[j] - (env.avg_weight[j] - env.dev_weight[j]) * vm[j]
            for j in range(n)) <= env.capacity, "capacity")

        # RC constraints 
        for k in range(env.K1):  
            model.addConstrs(
                (w1[k] - u1[k] + vp[j] - vm[j] == x[j]
                for j in env.group1[k]), "eq_x_1k_{}".format(k)
            )

        model.update()
        model.write("RC1.lp")
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
                z_sol = None
            else:
                print("No feasible solution found.")
                obj_val = None
                x_sol = None
                z_sol = None
        else :
            status = 'optimal'
            obj_val = model.ObjVal
            x_sol = np.array([i for i, var in x.items() if var.X > 0.5], dtype=int)
            z_sol = None
                 
        return status, obj_val, x_sol, z_sol, solve_time

    def RC2Gurobi(self, env, gp_env):
        n = env.N
        model = gp.Model("Compact Robust Counterpart of the Knapsack Problem", env = gp_env)
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self.tmax

        # decision variables
        x = model.addVars(n, vtype=GRB.BINARY, name="x")

        # level 1 variables
        w1 = model.addVars(env.K1, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w1")                     # w1[k] = max(0, e1[k])
        e1 = model.addVars(env.K1, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="e1")
    
        # objective function
        model.setObjective(gp.quicksum(env.value[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    
        # (robust) constraint
        model.addConstr(gp.quicksum((env.avg_weight[i] + env.dev_weight[i]) * x[i] for i in range(n))
                    + gp.quicksum(w1[k] for k in range(env.K1)) <= env.capacity, "capacity")
        
        # RC constraints
        for k in range(env.K1):
            model.addConstr(e1[k] == (env.avg_weight_1k[k] + env.dev_weight_1k[k]) 
                            - gp.quicksum((env.avg_weight[i] + env.dev_weight[i]) * x[i] for i in env.group1[k])
                            - gp.quicksum((env.avg_weight[i] - env.dev_weight[i]) * (1 -x[i]) for i in env.group1[k]), f"e1_def_{k}")
            model.addConstr(w1[k] == gp.min_(e1[k], 0), name=f"w1_def_{k}")
    
        t1 = time.perf_counter()   # start solve
        model.optimize()
        t2 = time.perf_counter()   # end solve
        solve_time = t2 - t1
    
        if model.status == GRB.OPTIMAL:
            status = 'optimal'
            best_value = model.objVal
            chosen_items = [i for i in range(n) if x[i].X > 0.5]

            return status, best_value, chosen_items, None, solve_time

        elif model.status == GRB.TIME_LIMIT:
            status = 'time_limit'

            return status, np.inf, None, None, solve_time  # Timeout

        else:
            status = 'infeasible'

            return status, np.inf, None, None, solve_time  # Infeasible