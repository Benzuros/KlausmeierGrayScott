import numpy as np
import matplotlib.pyplot as plt
from system import System


a = 2
u0 = lambda x, y: a
v0 = lambda x, y: np.random.rand()

S = System(20, 20, 0.1, 0.02, 20, 0.1, 0.45, a, u0, v0)
Fig, Ax = plt.subplots()
S.bifurcation_diagram(0.1, 10, log=True, ax=Ax)
Ax.legend()

plt.show()