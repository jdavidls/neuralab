#%%
from dataclasses import dataclass
import jax
import jax.numpy as np

from jax.numpy.linalg import eigh, inv, matrix_power
import numpy
from scipy.special import eval_legendre

from flax import linen as nn

#jax.scipy.special.


def init_A(N: int):
    def init1():
        P = np.sqrt(1 + 2 * np.arange(N, dtype=np.float64))
        A = P[:, np.newaxis] * P[np.newaxis, :]
        A = np.tril(A) - np.diag(np.arange(N))
        return -A
    
    def init2():
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        return A

    return init2

def init_B(N: int):
    def init1():
        B = np.sqrt(2 * np.arange(N) + 1.0)
        return B[:, None]

    def init2():
        q = np.arange(N, dtype=np.float64)
        T = np.sqrt(np.diag(2 * q + 1))
        B = np.diag(T)[:, None]
        B = B.copy()
        return B

    return init2

def init_C(N: int):
    def init():
        C = np.ones((1, N))
        return C
    return init

def init_D(N: int):
    def init():
        D = np.zeros((1,))
        return D
    return init

def init_leg(N: int, L: int):
    def init():
        vals = numpy.arange(0.0, L) / L
        eval_matrix = eval_legendre(numpy.arange(N)[:, None],  2 * vals - 1).T
        eval_matrix *= (2 * numpy.arange(N) + 1) ** .5 * (-1) ** numpy.arange(N)
        
        fn = lambda x: numpy.heaviside(x, 0.0) * numpy.heaviside(1.0 - x, 0.0)
        eval_matrix[fn(vals) == 0.0] = 0.0

        return eval_matrix
    return init

# self.vals = np.arange(0.0, T, dt)


def init_log_dt(dt_min=0.001, dt_max=0.1):
    def init(key):
        return jax.random.uniform(key, (1,)) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def discretize(A, B, C, step):
    I = np.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C

def ssm_x(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        #y_k = Cb @ x_k
        #return x_k, y_k
        return x_k, x_k
    

    return jax.lax.scan(step, x0, u)

from scipy.signal import cont2discrete

class HiPPO(nn.Module):
    N: int
    L: int


    def setup(self):
        N, L = self.N, self.L
        self.A = self.variable("constant", "A", init_A(N)).value
        self.B = self.variable("constant", "B", init_B(N)).value
        self.C = self.variable("constant", "C", init_C(N)).value
        self.D = self.variable("constant", "D", init_D(N)).value
        self.leg = self.variable("constant", "leg", init_leg(N, L)).value
        #self.dt = self.variable("cache", "dt", init_log_dt()).value
        
        #self.ABC, _, _ = cont2discrete((self.A, self.B, self.C, self.D), dt=1/L, method='bilinear')
        self.ABC = discretize(self.A, self.B, self.C, step=1/L)

        self.x_0 = self.variable("state", "x_0", np.zeros, (N,))

    def __call__(self, u):
        return self.encode(u)

    def encode(self, u):

        #x_0, y = ssm_x(*self.ABC, u[..., None], self.x_0.value)

        x_0, y = ssm_x(*self.ABC, u[..., None], self.x_0.value)

        if self.is_mutable_collection("state"):
            self.x_0.value = x_0
        return y

    def decode(self, y):
        v = self.leg @ y
        return v
        


import matplotlib.pyplot as plt



def generate_signal(T=1, dt=1/100, freq=3):
    """
    Generates a white noise signal.
    """
    nyquist_cutoff = 0.5 / dt
    if freq > nyquist_cutoff:
        raise ValueError(f"{freq} must not exceed the Nyquist frequency for the given dt ({nyquist_cutoff:0.3f})")

    n_coefficients = int(np.ceil(T / dt / 2.))
    shape = (n_coefficients + 1,)
    sigma = 0.5 * np.sqrt(0.5)
    coefficients = 1j * numpy.random.normal(0., sigma, size=shape)
    coefficients[..., -1] = 0.
    coefficients += numpy.random.normal(0., sigma, size=shape)
    coefficients[..., 0] = 0.

    set_to_zero = np.fft.rfftfreq(2 * n_coefficients, d=dt) > freq
    coefficients *= (1 - set_to_zero)
    power_correction = np.sqrt(1. - np.sum(set_to_zero, dtype=float) / n_coefficients)
    if power_correction > 0.:
        coefficients /= power_correction
    coefficients *= np.sqrt(2 * n_coefficients)
    signal = np.fft.irfft(coefficients, axis=-1)
    signal = signal - signal[..., :1]  # Start from 0
    return signal

L = 100
#u = np.cumsum(jax.random.normal(jax.random.PRNGKey(0), (L,)))
#u = generate_signal()

hippo = HiPPO(N=64, L=L)
params = hippo.init(jax.random.PRNGKey(0), u)
h = hippo.bind(params)

y = h(u)
print("leg", h.leg.shape)
print("y", y.shape)

v = h.decode(y[-1])

plt.plot(u)
plt.plot(np.flip(v))

#%%
def K_conv(Ab, Bb, Cb, L):
    return np.array(
        [(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
    )

K = K_conv(h.ABC[0], h.ABC[1], h.ABC[2], L)
