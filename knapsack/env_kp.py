import numpy as np
from scipy.spatial import distance_matrix

class KP_env:
    def __init__(self, N=100, R=100, alpha=0.5, K=None ,inst_num=0):
        self.N = N # number of items
        self.R = R # costs and weights are drawn uniformly from {1, ..., R}. 
        self.alpha = alpha # capacity is set to alpha * sum(weights), 0 < alpha < 1.
        self.K1 = K[0] # number of subclasses in the first level
        #self.K2 = K[1] # number of subclasses in the second level
        self.inst_num = inst_num

    def read_test_inst(self):

        rng = np.random.default_rng(self.inst_num)

        # Generate random values and weights for the knapsack problem
        self.value = rng.integers(1, self.R, size=self.N)
        self.avg_weight = rng.integers(1, self.R, size=self.N) # nominal values for the weights
        self.capacity = round(self.alpha * sum(self.avg_weight))
    
        max_dev = 20
        self.dev_weight = np.array([rng.integers(1, min(w, max_dev) + 1) for w in self.avg_weight]) # max deviation from the nominal value

        # definition of the one-level laminar strucure: partition of N into K equally sized sets
        group1_size = self.N // self.K1
        self.group1 = {k: list(range(k * group1_size, (k + 1) * group1_size)) for k in range(self.K1)}

        # nominal value of the budget for each set k, and max dev from the nominal 
        self.avg_weight_1k = {k: round(sum(self.avg_weight[i] for i in self.group1[k])) for k in range(self.K1)}
        self.dev_weight_1k = {k: int(0.05 * self.avg_weight_1k[k]) for k in range(self.K1)}