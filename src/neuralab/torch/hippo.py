#%%
"""HiPPO module for transition and reconstruction functionalities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import cont2discrete
from scipy import special as ss
import matplotlib.pyplot as plt



class HiPPO(nn.Module):
    """Linear time invariant x' = Ax + Bu."""

    def __init__(self, N, method='legt', dt=1.0, T=1.0, discretization='bilinear', scale=False, c=0.0):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.method = method
        self.N = N
        self.dt = dt
        self.T = T
        self.c = c

        A, B = self.transition(method, N)
        #A = A + np.eye(N) * c
        self.A = A
        self.B = B.squeeze(-1)
        #self.measure_fn = self.measure(method)

        C = np.ones((1, N))
        D = np.zeros((1,))
        dA, dB, _, _, _ = cont2discrete((A, B, C, D), dt=dt, method=discretization)

        dB = dB.squeeze(-1)

        self.register_buffer('dA', torch.Tensor(dA))  # (N, N)
        self.register_buffer('dB', torch.Tensor(dB))  # (N,)

        self.vals = np.arange(0.0, T, dt)
        self.eval_matrix = self.basis(self.method, self.N, self.vals, c=self.c)  # (T/dt, N)
        #self.measure = self.measure(self.method)(self.vals)

    def forward(self, inputs, fast=False):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """
        inputs = inputs.unsqueeze(-1)
        u = inputs * self.dB  # (length, ..., N)

        if fast:
            dA = self.dA.unsqueeze(0).expand(u.size(0), -1, -1)
            return self.variable_unroll_matrix(dA, u)

        c = torch.zeros(u.shape[1:]).to(inputs)
        cs = []
        for f in inputs:
            c = F.linear(c, self.dA) + self.dB * f
            cs.append(c)
        return torch.stack(cs, dim=0)


    @staticmethod
    def variable_unroll_matrix(A, u):
        """
        Unrolls the matrix multiplication for variable length sequences.
        A: (length, N, N)
        u: (length, ..., N)
        """
        c = torch.zeros_like(u[0])
        cs = []
        for t in range(len(u)):
            c = F.linear(c, A[t]) + u[t]
            cs.append(c)
        return torch.stack(cs, dim=0)
    

    def reconstruct(self, c, evals=None):
        """
        c: (..., N,) HiPPO coefficients (same as x(t) in S4 notation)
        output: (..., L,)
        """
        if evals is not None:
            eval_matrix = self.basis(self.method, self.N, evals)
        else:
            eval_matrix = self.eval_matrix

        #m = self.measure[self.measure != 0.0]

        c = c.unsqueeze(-1)
        y = eval_matrix.to(c) @ c
        return y.squeeze(-1).flip(-1)

    @staticmethod
    def transition(measure, N, **measure_args):
        if measure == 'lagt':
            b = measure_args.get('beta', 1.0)
            A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
            B = b * np.ones((N, 1))
        elif measure == 'legt':
            Q = np.arange(N, dtype=np.float64)
            R = (2 * Q + 1) ** .5
            j, i = np.meshgrid(Q, Q)
            A = R[:, None] * np.where(i < j, (-1.) ** (i - j), 1) * R[None, :]
            B = R[:, None]
            A = -A
        elif measure == 'legs':
            q = np.arange(N, dtype=np.float64)
            col, row = np.meshgrid(q, q)
            r = 2 * q + 1
            M = -(np.where(row >= col, r, 0) - np.diag(q))
            T = np.sqrt(np.diag(2 * q + 1))
            A = T @ M @ np.linalg.inv(T)
            B = np.diag(T)[:, None]
            B = B.copy()
        elif measure == 'fourier':
            freqs = np.arange(N // 2)
            d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
            A = 2 * np.pi * (-np.diag(d, 1) + np.diag(d, -1))
            B = np.zeros(N)
            B[0::2] = 2
            B[0] = 2 ** .5
            A = A - B[:, None] * B[None, :]
            B *= 2 ** .5
            B = B[:, None]
        return A, B

    @staticmethod
    def measure(method, c=0.0):
        if method == 'legt':
            fn = lambda x: np.heaviside(x, 0.0) * np.heaviside(1.0 - x, 0.0)
        elif method == 'legs':
            fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
        elif method == 'lagt':
            fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
        elif method in ['fourier']:
            fn = lambda x: np.heaviside(x, 1.0) * np.heaviside(1.0 - x, 1.0)
        else:
            raise NotImplementedError
        fn_tilted = lambda x: np.exp(c * x) * fn(x)
        return fn_tilted

    @staticmethod
    def basis(method, N, vals, c=0.0, truncate_measure=True):
        if method == 'legt':
            eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1).T
            eval_matrix *= (2 * np.arange(N) + 1) ** .5 * (-1) ** np.arange(N)
        elif method == 'legs':
            _vals = np.exp(-vals)
            eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * _vals).T
            eval_matrix *= (2 * np.arange(N) + 1) ** .5 * (-1) ** np.arange(N)
        elif method == 'lagt':
            vals = vals[::-1]
            eval_matrix = ss.eval_genlaguerre(np.arange(N)[:, None], 0, vals)
            eval_matrix = eval_matrix * np.exp(-vals / 2)
            eval_matrix = eval_matrix.T
        elif method == 'fourier':
            cos = 2 ** .5 * np.cos(2 * np.pi * np.arange(N // 2)[:, None] * (vals))
            sin = 2 ** .5 * np.sin(2 * np.pi * np.arange(N // 2)[:, None] * (vals))
            cos[0] /= 2 ** .5
            eval_matrix = np.stack([cos.T, sin.T], axis=-1).reshape(-1, N)

        if truncate_measure:
            eval_matrix[HiPPO.measure(method)(vals) == 0.0] = 0.0

        p = torch.tensor(eval_matrix)
        p *= np.exp(-c * vals)[:, None]
        return p


def generate_signal(T, dt, freq):
    """
    Generates a white noise signal.
    """
    nyquist_cutoff = 0.5 / dt
    if freq > nyquist_cutoff:
        raise ValueError(f"{freq} must not exceed the Nyquist frequency for the given dt ({nyquist_cutoff:0.3f})")

    n_coefficients = int(np.ceil(T / dt / 2.))
    shape = (n_coefficients + 1,)
    sigma = 0.5 * np.sqrt(0.5)
    coefficients = 1j * np.random.normal(0., sigma, size=shape)
    coefficients[..., -1] = 0.
    coefficients += np.random.normal(0., sigma, size=shape)
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

def plot_hippo_reconstruction(T, dt, N, freq):
    """
    Plots the original signal and its HiPPO reconstruction.
    """
    vals = np.arange(0.0, T, dt)
    signal = generate_signal(T, dt, freq)
    signal = torch.tensor(signal, dtype=torch.float)

    plt.figure(figsize=(16, 8))
    plt.plot(vals, signal.numpy(), 'k', linewidth=1.0, label='Original Signal')

    methods = ['legt', 'fourier', 'legs']
    for method in methods:
        hippo = HiPPO(N=N, method=method, dt=dt, T=T)
        hippo_projection = hippo(signal)
        reconstructed_signal = hippo.reconstruct(hippo_projection[-1]).numpy()
        plt.plot(vals, reconstructed_signal, label=f'HiPPO {method}')

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('HiPPO Reconstruction')
    plt.show()

    plt.plot(hippo_projection[-1])
    plt.show()


if __name__ == '__main__':
    plot_hippo_reconstruction(T=2, dt=1e-2, N=64, freq=3.0)
# %%
