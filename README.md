# Signed Budget Uncertainty for Robust Mixed-Integer Optimization

Implementation of the methods presented in the paper "Signed Budget Uncertainty for Robust Mixed-Integer Optimization":
- Compact robust counterparts for (laminar) signed budget uncertainty sets with binary uncertainty affected variables.
- Enumeration of $2^K$ nominal problems.

The methods are tested on maximum coverage and knapsack instances.

### Main Dependencies Installation

In order to execute the code, you need an [Anaconda](https://www.anaconda.com/) environment with Python>=3.10.

For the packages installation, open a terminal (Anaconda Prompt for Windows users) in the project root folder and execute the following commands.

```
pip install joblib
pip install datetime
pip install gurobipy
pip install numpy
pip install copy
pip install psutil
pip install time
pip install tabulate
pip install os
pip install scipy
```

### Usage

Run file ```main_MCP.py``` to test the following methods on maximum coverage instances:

- ```nominal```: solves the deterministic problem;
- ```RC1``` : classical robust counterpart derived from duality for signed budget uncertainty;
- ```RC2``` : compact robust counterpart for signed budget uncertainty (explicit big-M formulation);
- ```RC2Gurobi```: compact robust counterpart for signed budget uncertainty (uses Gurobi's built-in functions).

 Run file ```main_MCP_L2.py``` to test above methods on maximum coverage instances under two-level laminar budget uncertainty.

 Run file ```cRC/main_cMCP.py``` to compare signed budget uncertainty (```RC2Gurobi```) and classical budget uncertainty (```cRC```) for different values of $\Gamma$ on maximum coverage instances.

 Run file ```knapsack/main_kp.py``` to compare the compact robust counterpart (```RC2Gurobi```) and the enumeration of $2^K$ nominal problems (```2K_nominal```) on knapsack instances with constraint one-level laminar signed budget uncertainty.

 Instances are defined in the files ```environment.py```.

