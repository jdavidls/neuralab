#%%
import jax
import jax.numpy as np
from flax import nnx
from jax.numpy.linalg import inv
from scipy.special import eval_legendre


def make_hippo(N: int):
    n = np.arange(N)
    P = np.sqrt(1 + 2 * n)

    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(n)

    B = np.sqrt(2 * n + 1.0)
    B = B[:, None]

    return -A, B

def discretize(A, B, step):
    I = np.eye(A.shape[0])
    BL = inv(I - (step / 2) * A)
    Ab = BL @ (I + (step / 2) * A)
    Bb = (BL * step) @ B
    return Ab, Bb


def hippo_basis(N: int, L: float, dt:float):
    n = np.arange(N)
    l = np.arange(L)
    t = np.exp(-l * dt)
    basis = eval_legendre(n[:, None], 1 - 2 * t).T
    basis = basis * (2 * n + 1) ** .5 * (-1) ** n
    return basis


def init_log_dt(dt_min=0.001, dt_max=0.1):
    def init(key):
        return jax.random.uniform(key, (1,)) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def hippo_scan(Ab, Bb, x_z, u, unroll=128):
    def hippo_step(x0, u_k):
        x1 = Ab @ x0 + Bb @ u_k
        return x1, x1

    return jax.lax.scan(hippo_step, x_z, u, unroll=unroll)

class HiPPO(nnx.Module):

    def __init__(self, N: int, L: int, dt: float, rngs: nnx.Rngs):
        A, B = make_hippo(N)

        Ab, Bb = discretize(A, B, dt)
        self.Ab = nnx.Cache(Ab)
        self.Bb = nnx.Cache(Bb)

        basis = hippo_basis(N, L, dt)
        self.basis = nnx.Cache(basis) ## usea a creation_hook

        self.last_x = nnx.Variable(np.zeros(N,))

    def encode(self, u, update_state = False):
        Ab, Bb = self.Ab, self.Bb
        
        x_N, f = hippo_scan(Ab, Bb, self.last_x.value, u[..., None])

        if update_state:
            self.last_x.value = x_N

        return x

    def decode(self, x = None):
        basis = self.basis

        if x is None:
            x = self.last_x.value

        return basis @ x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    L = 512
    u = np.cumsum(jax.random.normal(jax.random.PRNGKey(3), (L,)))


    h = HiPPO(N=64, L=L, dt=1/L)
    x = h.encode(u)
    OFFSET = L-1
    v = h.decode(x[OFFSET])

    plt.plot(u)
    plt.plot(np.flip(v)[-OFFSET:])
    plt.show()

    plt.pcolormesh(x)
    plt.colorbar()
    plt.show()

