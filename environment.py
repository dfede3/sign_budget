import numpy as np
import os
from scipy.spatial import distance_matrix
import copy
        
class RMCP_env:
    def __init__(self, I, r, inst_num = 0):
        self.I = I # number of base locations
        self.J = I # number of demand locations
        self.r = r # coverage radius of a facility
        self.inst_num = inst_num

    def make_test_inst(self, save_env = True):
        env = copy.deepcopy(self)
        rng = np.random.default_rng(env.inst_num)
        env.base_loc = rng.random((env.I, 2)) # I random location points in the unit square
        env.dem_loc  = copy.deepcopy(env.base_loc) # base and demand locations coincide

        env.dem = rng.integers(1, 15, size=env.J) # nominal demand values
        max_dev = 10
        env.dev = np.array([rng.integers(1, min(d, max_dev) + 1) for d in env.dem]) # maximum absolute deviation allowed from the nominal value
        env.Dem = sum(env.dem) # global budget
        env.Dev = 0 # max deviation allowed from the global budget

        # Building the distance matrix
        env.N = {j: [] for j in range(env.J)}
        block = 500  # max number of lines to processed 
        for i_start in range(0, env.I, block):
            i_end = min(i_start + block, env.I)
            D_block = distance_matrix(env.base_loc[i_start:i_end], env.dem_loc)  # (block, J)
            rows, cols = np.where(D_block <= env.r)
            for local_i, j in zip(rows, cols):
                env.N[j].append(i_start + local_i)
            
        if save_env:
            env.write_test_inst()
    
    def write_test_inst(self):
        os.makedirs("data/rmcp", exist_ok=True) 
        inst_path = f"data/rmcp/rmcp_env_I{self.I}_" \
                    f"J{self.J}_" \
                    f"r{self.r}_" \
                    f"i{self.inst_num}.txt"
        f = open(inst_path, "w+")
        f.write(" ".join(self.base_loc[:, 0].astype(str)) + "\n")
        f.write(" ".join(self.base_loc[:, 1].astype(str)) + "\n")
        f.write(" ".join(self.dem_loc[:, 0].astype(str)) + "\n")
        f.write(" ".join(self.dem_loc[:, 1].astype(str)) + "\n")
        f.write(" ".join(self.dem.astype(str)) + "\n")
        f.write(" ".join(self.dev.astype(str)) + "\n")
        f.write(str(self.Dem) + "\n")
        f.write(str(self.Dev) + "\n")
        for j in range(self.J):
            f.write(" ".join(map(str, self.N[j])) + "\n")
        f.close()

    def read_test_inst(self):
        inst_path = f"data/rmcp/rmcp_env_I{self.I}_" \
                    f"J{self.J}_" \
                    f"r{self.r}_" \
                    f"i{self.inst_num}.txt"
        
        with open(inst_path, 'r') as f:
            f_lines = [line.strip() for line in f.readlines()]
        
        # 1–2: base locations
        base_x = np.array(f_lines[0].split(), dtype=float)
        base_y = np.array(f_lines[1].split(), dtype=float)
        self.base_loc = np.vstack((base_x, base_y)).T

        # 3–4: demand locations
        dem_x = np.array(f_lines[2].split(), dtype=float)
        dem_y = np.array(f_lines[3].split(), dtype=float)
        self.dem_loc = np.vstack((dem_x, dem_y)).T

        # 5: demand 
        self.dem = np.array(f_lines[4].split(), dtype=int)
        self.dev = np.array(f_lines[5].split(), dtype=int)
        self.Dem = float(f_lines[6])
        self.Dev = float(f_lines[7])

        self.N = {}
        for j, line in enumerate(f_lines[8:]):
            self.N[j] = list(map(int, line.split())) if line else []

class RMCP_L2_env:
    def __init__(self, I, r, inst_num = 0):
        self.I = I # number of base locations
        self.J = I # number of demand locations
        self.r = r # coverage radius of a facility
        self.inst_num = inst_num

    def make_test_inst(self, save_env = True):
        env = copy.deepcopy(self)
        rng = np.random.default_rng(env.inst_num)
        env.base_loc = rng.random((env.I, 2))
        env.dem_loc  = copy.deepcopy(env.base_loc)

        env.dem = rng.integers(1, 15, size=env.J)
        max_dev = 10
        env.dev = np.array([rng.integers(1, min(d, max_dev) + 1) for d in env.dem])

        # Building the distance matrix N
        env.N = {j: [] for j in range(env.J)}
        block = 500 
        for i_start in range(0, env.I, block):
            i_end = min(i_start + block, env.I)
            D_block = distance_matrix(env.base_loc[i_start:i_end], env.dem_loc)  # (block, J)
            rows, cols = np.where(D_block <= env.r)
            for local_i, j in zip(rows, cols):
                env.N[j].append(i_start + local_i)

        if save_env:
            env.write_test_inst()
    
    def write_test_inst(self):
        os.makedirs("data/rmcp", exist_ok=True) 
        inst_path = f"data/rmcp/rmcp_env_I{self.I}_" \
                    f"J{self.J}_" \
                    f"r{self.r}_" \
                    f"i{self.inst_num}.txt"
        f = open(inst_path, "w+")
        f.write(" ".join(self.base_loc[:, 0].astype(str)) + "\n")
        f.write(" ".join(self.base_loc[:, 1].astype(str)) + "\n")
        f.write(" ".join(self.dem_loc[:, 0].astype(str)) + "\n")
        f.write(" ".join(self.dem_loc[:, 1].astype(str)) + "\n")
        f.write(" ".join(self.dem.astype(str)) + "\n")
        f.write(" ".join(self.dev.astype(str)) + "\n")
        for j in range(self.J):
            f.write(" ".join(map(str, self.N[j])) + "\n")
        f.close()

    def read_test_inst(self):
        inst_path = f"data/rmcp/rmcp_env_I{self.I}_" \
                    f"J{self.J}_" \
                    f"r{self.r}_" \
                    f"i{self.inst_num}.txt"
        
        with open(inst_path, 'r') as f:
            f_lines = [line.strip() for line in f.readlines()]
        
        # 1–2: base locations
        base_x = np.array(f_lines[0].split(), dtype=float)
        base_y = np.array(f_lines[1].split(), dtype=float)
        self.base_loc = np.vstack((base_x, base_y)).T

        # 3–4: demand locations
        dem_x = np.array(f_lines[2].split(), dtype=float)
        dem_y = np.array(f_lines[3].split(), dtype=float)
        self.dem_loc = np.vstack((dem_x, dem_y)).T

        # 5: demand 
        self.dem = np.array(f_lines[4].split(), dtype=int)
        self.dev = np.array(f_lines[5].split(), dtype=int)

        self.N = {}
        for j, line in enumerate(f_lines[6:]):
            self.N[j] = list(map(int, line.split())) if line else []
        
        self.unc = UncertaintySet_L2(self)
        # Dem[2][1] = sum of all nominal demands (global budget in the second level)
        self.Dem = {1: {}, 2: {}}
        self.Dem[2][1] = int(sum(self.dem))

        # Dem[1][k] = sum of the nominal demands in group (1,k)
        for k in range(1, self.unc.K[1] + 1):
            members = self.unc.get_group(1, k)
            self.Dem[1][k] = int(sum(self.dem[j] for j in members))

        self.Dev = {1: {}, 2: {}}
        for k in range(1, self.unc.K[1] + 1):
            self.Dev[1][k] = int(0.05 * self.Dem[1][k]) # max deviation allowed from Dem[1][k]

        self.Dev[2][1] = int(0.01 * self.Dem[2][1]) # max deviation allowed from Dem[2][1]


class UncertaintySet_L2:

    def __init__(self, env):
        self.n1 = env.I  # number of facility/base locations
        self.L  = 2 # number of levels in the laminar family
        self.P  = self._build_partition(env)
        self.K  = {1: 4, 2: 1}  # K[l]: number of groups in level l

    def _build_partition(self, env):
        P = {1: {}, 2: {}}

        # Level 1: four quadrants based on base_loc coordinates
        # Quadrant 1: [0, 0.5] x [0, 0.5]  (bottom-left)
        # Quadrant 2: (0.5, 1] x [0, 0.5]  (bottom-right)
        # Quadrant 3: [0, 0.5] x (0.5, 1]  (top-left)
        # Quadrant 4: (0.5, 1] x (0.5, 1]  (top-right)
        x = env.base_loc[:, 0]
        y = env.base_loc[:, 1]

        P[1][1] = list(np.where((x <= 0.5) & (y <= 0.5))[0])
        P[1][2] = list(np.where((x >  0.5) & (y <= 0.5))[0])
        P[1][3] = list(np.where((x <= 0.5) & (y >  0.5))[0])
        P[1][4] = list(np.where((x >  0.5) & (y >  0.5))[0])

        # Level 2: all indices
        P[2][1] = list(range(self.n1))

        return P

    def get_group(self, l, k):
        # Return list of indices in group k of level l
        return self.P[l][k]

    def find_groups(self, j):
        
         # Return all (l, k) such that j in P[l][k].
        
        groups = []
        for l in range(1, self.L + 1):
            for k in range(1, self.K[l] + 1):
                if j in self.P[l][k]:
                    groups.append((l, k))
        return groups