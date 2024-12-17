#%%
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp


def legendre(N, t):
    def coef(n):
        coef = [0] * N
        coef[n] = 1
        return coef

    return np.array([
        np.polynomial.legendre.Legendre(coef(n))(t) for n in range(N)
    ])
        

N = 8
L = 128
w = np.linspace(-1, 0, L)
K = legendre(N, w)
#W = np.sum(K, axis=1)
#K = K / W

u = np.cumsum(np.random.normal(size=1024))

f = np.array([np.convolve(u, K[n], 'valid') for n in range(N)]) / L
# %%
n = 200
ir = f[:, n]
y = np.sum(ir[:, None] * K, axis=0) / N

u_ = u[L-1+n:L-1+n+L]

#plt.plot(u_)
plt.plot(y)

# %%
