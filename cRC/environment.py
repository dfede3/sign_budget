import numpy as np
import os
from scipy.spatial import distance_matrix
import copy
        
class cRMCP_env:
    def __init__(self, I, r, inst_num = 0):
        self.I = I # number of base locations
        self.J = I # number of demand locations
        self.r = r # coverage radius a facility
        self.inst_num = inst_num

    def make_test_inst(self, save_env = True):
        env = copy.deepcopy(self)
        rng = np.random.default_rng(env.inst_num)
        env.base_loc = rng.random((env.I, 2))
        env.dem_loc  = copy.deepcopy(env.base_loc)

        env.dem = rng.integers(1, 15, size=env.J)
        max_dev = 10
        env.dev = np.array([rng.integers(1, min(d, max_dev) + 1) for d in env.dem])
        env.Dem = env.J # Gamma = J
        env.Dev = 0

        # Building the distance matrix
        env.N = {j: [] for j in range(env.J)}
        block = 500  
        for i_start in range(0, env.I, block):
            i_end = min(i_start + block, env.I)
            D_block = distance_matrix(env.base_loc[i_start:i_end], env.dem_loc)  
            rows, cols = np.where(D_block <= env.r)
            for local_i, j in zip(rows, cols):
                env.N[j].append(i_start + local_i)

        if save_env:
            env.write_test_inst()
    
    def write_test_inst(self):
        os.makedirs("data/CB", exist_ok=True) 
        inst_path = f"data/CB/rmcp_env_I{self.I}_" \
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
        inst_path = f"data/CB/rmcp_env_I{self.I}_" \
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
        self.Dem = float(f_lines[6]) # classical budget
        self.Dev = float(f_lines[7])
        self.DemSB = sum(self.dem) # signed budget
        self.DevSB = 0

        self.N = {}
        for j, line in enumerate(f_lines[8:]):
            self.N[j] = list(map(int, line.split())) if line else []