import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags_array, eye_array, kron
from scipy.sparse.linalg import factorized
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
        self.u = np.zeros((1, len(self.y), len(self.x)))
        self.u[0] = u0(mx, my)
        self.v = np.zeros((1, len(self.y), len(self.x)))
        self.v[0] = v0(mx, my)

        # discrete laplacian matrix for simulations
        nx = len(self.x) - 2
        ny = len(self.y) - 2
        lap1 = diags_array([1., -4., 1.], offsets=(-1, 0, 1), shape=(nx, nx))
        lap = kron(eye_array(ny), lap1) \
              + kron(diags_array([1., 1.], offsets=(-1, 1), shape=(ny, ny)), eye_array(nx))
        self.laplacian = lap / self.h ** 2


    def change_parameters(self, **kwargs):
        """Change the parameters of the system given in kwargs.
        You may only change the following parameters: d1, d2, m, a."""
        if kwargs.keys() - {"d1", "d2", "m", "a"}:
            raise KeyError(f"Parameters {kwargs.keys() - {"d1", "d2", "m", "a"}} not allowed. "
                             "You may only change the following parameters: d1, d2, m, a.")
        if "d1" in kwargs.keys():
            self.d1 = kwargs["d1"]
        if "d2" in kwargs.keys():
            self.d2 = kwargs["d2"]
        if "m" in kwargs.keys():
            self.m = kwargs["m"]
        if "a" in kwargs.keys():
            self.a = kwargs["a"]


    def clear_history(self):
        """Clear the history of the system. u and v will now only consist of their current state (u[-1], v[-1])."""
        self.u = self.u[-1:, :, :]
        self.v = self.v[-1:, :, :]
        self.current_time = 0



    def simulate(self, time, extend=False):
        """Simulate the system for the given time, beginning from the current state (u[-1], v[-1]).
        If extend is False (default), change self.u and self.v to the results of this simulation.
        If Extend is True, treat the simulation as a continuation of the previous one:
        Extend self.u and self.v and self.current_time by the results
        (useful for analyzing the evolution of the system over multiple calls of this method).
        Return the max and mean of v[-1]."""
        t = np.arange(0, time + self.ht, self.ht)
        nt, nx, ny = len(t), len(self.x), len(self.y)
        u = np.zeros((nt, ny, nx))
        v = np.zeros((nt, ny, nx))
        u[0] = self.u[-1]
        v[0] = self.v[-1]
        lap = self.laplacian

        # create matrix for LHS
        au = (eye_array(lap.shape[0]) - self.ht * self.d1 * lap).tocsc()
        av = (eye_array(lap.shape[0]) - self.ht * self.d2 * lap).tocsc()

        # LU decomposition
        solve_u = factorized(au)
        solve_v = factorized(av)



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
            u_new = solve_u(rhs_u)
            v_new = solve_v(rhs_v)

            # add solution
            u[n + 1, 1:-1, 1:-1] = u_new.reshape((ny - 2, nx - 2))
            v[n + 1, 1:-1, 1:-1] = v_new.reshape((ny - 2, nx - 2))

            # boundary
            u[n + 1, :, 0] = u[n + 1, :, -1] = 0
            u[n + 1, 0, :] = u[n + 1, -1, :] = 0
            v[n + 1, :, 0] = v[n + 1, :, -1] = 0
            v[n + 1, 0, :] = v[n + 1, -1, :] = 0

        if extend:
            self.u = np.concatenate((self.u, u[1:]))
            self.v = np.concatenate((self.v, v[1:]))
            self.current_time += time
        else:
            self.u = u
            self.v = v
            self.current_time = time

        return np.max(self.v[-1]), np.mean(self.v[-1])


    def simulate_until_stable(self, step, tolerance=0.001, log=False, extend=True):
        """Repeatedly call self.simulate(step, extend) until the Frobenius (L^(2,2)) norm of
        v[-1] - v[-2] is less than tolerance.
        If log is True, print the current mean and max of v after every iteration and the total number of iterations
        at the end. Note that extend is True by default.
        If extend is False, u and v will only cover the last iteration."""
        i = 1
        v_max, v_mean = self.simulate(step, extend)
        if log:
            print(f"max: {v_max:.3f}, mean: {v_mean:.3f}")

        while np.linalg.norm(self.v[-1] - self.v[-2]) >= tolerance:
            v_max, v_mean = self.simulate(step, extend)
            i += 1
            if log:
                print(f"max: {v_max:.3f}, mean: {v_mean:.3f}")

        if log:
            print(f"Reached stability after {i} iterations.")
            print(f"Current max  of v: {v_max:.3f}")
            print(f"Current mean of v: {v_mean:.3f}")

        return v_max, v_mean


    def bifurcation_diagram(self, a_step, t_step, tolerance=0.001, log=False, ax=None):
        """Compute the bifurcation diagram of the system using numerical continuation.
        Return two lists: max and mean, consisting of tuples (a, max) and (a, mean) respectively.
        If ax is given, plot both max and mean as a scatter plot on ax."""
        maxs = []
        means = []
        a = a_old = self.a
        u_st = np.array([])
        v_st = np.array([])

        while a > 0:
            ma, me = self.simulate_until_stable(t_step, tolerance, log, extend=False)
            if ma >= tolerance:
                u_st = self.u[-1]
                v_st = self.v[-1]
            maxs.append((a, ma))
            means.append((a, me))
            self.change_parameters(a = a - a_step)
            a = self.a
            if log:
                print(f"a = {a}")

        while a <= a_old:
            ma, me = self.simulate_until_stable(t_step, tolerance, log, extend=False)
            if ma < tolerance:
                self.u[-1] = u_st
                self.v[-1] = v_st
            else:
                u_st = self.u[-1]
                v_st = self.v[-1]
            maxs.append((a, ma))
            means.append((a, me))
            self.change_parameters(a = a + a_step)
            a = self.a
            if log:
                print(f"a = {a}")

        if ax:
            ax.set(xlabel="$a$", ylabel="stationary states $v^*$")
            ax.scatter(*zip(*maxs), color="blue", label="max($v^*$)")
            ax.scatter(*zip(*means), color="red", label="mean($v^*$)")

        return maxs, means


    def plot(self, fig, ax, t=-1, u=False):
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
        cb = fig.colorbar(pt)
        ax.set_title(f"$a$ = {self.a:.2f}, $m$ = {self.m:.2f}")

        return pt, cb


    def animate(self, fig, ax, u=False):
        """Return a matplolib FuncAnimation of the evolution of v (u if u=True)."""
        if u:
            p = self.u
            cm = "YlGnBu"
        else:
            p = self.v
            cm = "YlGn"

        pt = ax.imshow(p[0], origin="lower", extent=(0, self.xmax, 0, self.ymax), cmap=cm, clim=(0, np.max(p)))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        txt = ax.text(0.5, 0.95, "$t = 0$", transform=ax.transAxes, fontsize=14,
                      verticalalignment='top', bbox=props)
        fig.colorbar(pt, ax=ax)

        def update(frame):
            pt.set_data(p[frame])
            txt.set_text(f"$t = {frame/len(p) * self.current_time :n}$")
            return pt, txt

        anim = FuncAnimation(fig, update, frames=len(p), interval=10, blit=True)

        return anim




if __name__ == '__main__':
    uu = lambda x, y: 1.2
    vv = lambda x, y: np.random.rand()

    S = System(20, 20, 0.1, 0.1, 10, 0.1, 0.4, 1.2, uu, vv)
    S.simulate_until_stable(100, log=True, extend=False)
    Fig, Ax = plt.subplots()
    S.plot(fig=Fig, ax=Ax)
    # Fig, Ax = plt.subplots()
    # ani = S.animate(fig=Fig, ax=Ax)

    plt.show()