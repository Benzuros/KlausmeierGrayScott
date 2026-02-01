import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


class System:
    def __init__(self, xmax, ymax, h, ht, d1, d2, m, a, u0, v0):
        self.xmax = xmax
        self.ymax = ymax
        self.h = h
        self.ht = ht
        self.d1 = d1
        self.d2 = d2
        self.m = m
        self.a = a

        self.current_time = 0
        self.x = np.arange(0, xmax + h, h)
        self.y = np.arange(0, ymax + h, h)

        u0 = np.vectorize(u0)
        v0 = np.vectorize(v0)
        mx, my = np.meshgrid(self.x, self.y)
        self.u = np.zeros((1, len(self.y), len(self.x)), dtype=np.float64)
        self.u[0] = u0(mx, my)
        self.v = np.zeros((1, len(self.y), len(self.x)), dtype=np.float64)
        self.v[0] = v0(mx, my)


    def simulate(self, time):
        """Simulate the system for the given time, beginning from the current state (u[-1], v[-1]).
        Extend u and v appropriately."""
        t = np.arange(0, time + self.ht, self.ht)
        u = np.zeros((len(t), len(self.y), len(self.x)), dtype=np.float64)
        v = np.zeros((len(t), len(self.y), len(self.x)), dtype=np.float64)
        for n in range(len(self.u)):
            u[n] = self.u[n]
            v[n] = self.v[n]

        for n in tqdm(range(len(t) - 1)):
            #interior
            u[n + 1, 1:-1, 1:-1] = (u[n, 1:-1, 1:-1] + self.d1 * self.ht / self.h**2 * (u[n, 1:-1, 2:] + u[n, 1:-1, :-2]
                + u[n, 2:, 1:-1] + u[n, :-2, 1:-1] - 4 * u[n, 1:-1, 1:-1])
                + self.ht * (self.a - u[n, 1:-1, 1:-1] + u[n, 1:-1, 1:-1] * v[n, 1:-1, 1:-1]**2))
            v[n + 1, 1:-1, 1:-1] = (v[n, 1:-1, 1:-1] + self.d1 * self.ht / self.h**2 * (v[n, 1:-1, 2:] + v[n, 1:-1, :-2]
                + v[n, 2:, 1:-1] + v[n, :-2, 1:-1] - 4 * v[n, 1:-1, 1:-1])
                + self.ht * (u[n, 1:-1, 1:-1] * v[n, 1:-1, 1:-1]**2 - self.m * v[n, 1:-1, 1:-1]))
            #boundary
            u[n + 1, 1:-1, 0] = v[n + 1, 1:-1, 0] = np.zeros(len(self.y) - 2)
            u[n + 1, 1:-1, len(self.x) - 1] = v[n + 1, 1:-1, len(self.x) - 1] = np.zeros(len(self.y) - 2)
            u[n + 1, 0, :] = v[n + 1, 0, :] = np.zeros(len(self.x))
            u[n + 1, len(self.y) - 1, :] = v[n + 1, len(self.y) - 1, :] = np.zeros(len(self.x))

        self.current_time = time
        self.u = u
        self.v = v

        return self.u, self.v


    def simulate_until_stable(self, step):
        """Simulate the system for time step repeatedly until u[-1] = u[-2] and v[-1] = v[-2]."""
        while self.u[-1] != self.u[-2] and self.v[-1] != self.v[-2]:
            self.simulate(step)

        return self.u, self.v


    def plot(self, ax, t=-1, u=False):
        """Plot v[-1] on axes ax. If u is True, plot u[-1] instead."""
        if t == -1:
            n = -1
        else:
            n = t // self.ht
        if u:
            p = self.u[n]
        else:
            p = self.v[n]

        pt = ax.imshow(p, origin="lower", cmap="YlGn", extent=(0, self.xmax, 0, self.ymax))

        return pt


    def animate(self, fig, ax, u=False):
        if u:
            p = self.u
        else:
            p = self.v

        pt = ax.imshow(p[0], origin="lower", cmap="YlGn", extent=(0, self.xmax, 0, self.ymax))
        cb = fig.colorbar(pt, ax=ax)

        def update(frame):
            pt.set_data(p[frame])
            cb.update_normal(pt)
            return pt,

        anim = FuncAnimation(fig, update, frames=len(p), interval=50, blit=True)

        return anim




if __name__ == '__main__':
    def vv(x, y):
        if (x - 5)**2 + (y - 5)**2 <=  1:
            return 3
        return 0

    S = System(30, 30, 0.1, 0.001, 80, 1, 0.45, 1, lambda x, y: 1, vv)
    S.simulate(1)
    Fig, Ax = plt.subplots()
    Anim = S.animate(Fig, Ax)

    plt.show()