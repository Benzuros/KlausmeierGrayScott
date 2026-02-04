import numpy as np
import matplotlib.pyplot as plt
from system import System


v0 = lambda x, y: np.random.rand()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, layout="constrained")

s1 = System(20, 20, 0.1, 0.1, 10, 0.1, 0.45, 1.15, lambda x, y: 1.15, v0)
s1.simulate_until_stable(100, extend=False)
s1.plot(fig=fig, ax=ax1)
s2 = System(20, 20, 0.1, 0.1, 10, 0.1, 0.45, 1.2, lambda x, y: 1.2, v0)
s2.simulate_until_stable(100, extend=False)
s2.plot(fig=fig, ax=ax2)
s3 = S = System(20, 20, 0.1, 0.1, 10, 0.1, 0.45, 1.4, lambda x, y: 1.4, v0)
s3.simulate_until_stable(100, extend=False)
s3.plot(fig=fig, ax=ax3)
s4 = System(20, 20, 0.1, 0.1, 10, 0.1, 0.5, 1.25,  lambda x, y: 1.25, v0)
s4.simulate_until_stable(100, extend=False)
s4.plot(fig=fig, ax=ax4)

plt.show()
