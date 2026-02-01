import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags_array, eye_array, kron
from scipy.sparse.linalg import spsolve
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

        # discrete laplacian matrix for simulations
        nx = len(self.x) - 2
        ny = len(self.y) - 2
        lap1 = diags_array([1., -4., 1.], offsets=(-1, 0, 1), shape=(nx, nx))
        lap = kron(eye_array(ny), lap1) \
              + kron(diags_array([1., 1.], offsets=(-1, 1), shape=(ny, ny)), eye_array(nx))
        self.laplacian = lap / self.h ** 2



    def simulate(self, time):
        """Simulate the system for the given time, beginning from the current state (u[-1], v[-1]).
        Extend u and v appropriately."""
        t = np.arange(0, time + self.ht, self.ht)
        nt, nx, ny = len(t), len(self.x), len(self.y)
        u = np.zeros((nt, ny, nx))
        v = np.zeros((nt, ny, nx))
        u[0] = self.u[-1]
        v[0] = self.v[-1]
        lap = self.laplacian

        # create matrix for LHS
        au = eye_array(lap.shape[0]) - self.ht * self.d1 * lap
        av = eye_array(lap.shape[0]) - self.ht * self.d2 * lap


        for n in tqdm(range(nt - 1)):
            u_n = u[n]
            v_n = v[n]

            # reaction (explicit)
            ru = self.a - u_n[1:-1,1:-1] - u_n[1:-1,1:-1] * v_n[1:-1,1:-1]**2
            rv = u_n[1:-1,1:-1] * v_n[1:-1,1:-1]**2 - self.m * v_n[1:-1,1:-1]

            # RHS
            rhs_u = u_n[1:-1, 1:-1].ravel() + self.ht * ru.ravel()
            rhs_v = v_n[1:-1, 1:-1].ravel() + self.ht * rv.ravel()

            # diffusion (implicit)
            u_new = spsolve(au, rhs_u)
            v_new = spsolve(av, rhs_v)

            # add solution
            u[n + 1, 1:-1, 1:-1] = u_new.reshape((ny - 2, nx - 2))
            v[n + 1, 1:-1, 1:-1] = v_new.reshape((ny - 2, nx - 2))

            # boundary
            u[n + 1, :, 0] = u[n + 1, :, -1] = 0
            u[n + 1, 0, :] = u[n + 1, -1, :] = 0
            v[n + 1, :, 0] = v[n + 1, :, -1] = 0
            v[n + 1, 0, :] = v[n + 1, -1, :] = 0

        self.u = np.concatenate((self.u, u[1:]))
        self.v = np.concatenate((self.v, v[1:]))
        self.current_time += time

        return self.u, self.v


    def simulate_until_stable(self, step, tolerance=0.0001):
        """Simulate the system for time step repeatedly until the frobenius (L^2,2) norms of
        u[-1] - u[-2] and v[-1] - v[-2] are both less than tolerance."""
        while max(np.linalg.norm(self.u[-1] - self.u[-2]), np.linalg.norm(self.v[-1] - self.v[-2])) >= tolerance:
            self.simulate(step)

        return self.u, self.v


    def plot(self, ax, t=-1, u=False):
        """Plot v[t // ht] on axes ax. If u is True, plot u[t // ht] instead."""
        if t == -1:
            n = -1
        else:
            n = t // self.ht
        if u:
            p = self.u[n]
            cm = "YlGnBu"
        else:
            p = self.v[n]
            cm = "YlGn"

        pt = ax.imshow(p, origin="lower", cmap=cm, extent=(0, self.xmax, 0, self.ymax))

        return pt


    def animate(self, fig, ax, u=False):
        """Return a matplolib FuncAnimation of the evolution of v (u if u=True)."""
        if u:
            p = self.u
            cm = "YlGnBu"
        else:
            p = self.v
            cm = "YlGn"

        pt = ax.imshow(p[0], origin="lower", cmap=cm, extent=(0, self.xmax, 0, self.ymax))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        txt = ax.text(0.5, 0.95, "$t = 0$", transform=ax.transAxes, fontsize=14,
                      verticalalignment='top', bbox=props)
        fig.colorbar(pt, ax=ax)

        def update(frame):
            pt.set_data(p[frame])
            txt.set_text(f"$t = {frame/len(p) * self.current_time :.1f}$")
            return pt, txt

        anim = FuncAnimation(fig, update, frames=len(p), interval=10, blit=True)

        return anim




if __name__ == '__main__':
    A = 2
    uu = lambda x, y: A
    vv = lambda x, y: np.random.rand()

    S = System(20, 20, 0.1, 0.02, 20, 0.1, 0.45, A, uu, vv)
    S.simulate(20)
    Fig, Ax = plt.subplots()
    P = S.plot(Ax)
    Fig.colorbar(P, ax=Ax)
    Fig, Ax = plt.subplots()
    Anim = S.animate(Fig, Ax)

    plt.show()