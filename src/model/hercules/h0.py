# %%
import optax
from einops import rearrange
from flax import nnx
from jax import lax
from jax import numpy as np
from jax import random
from jaxtyping import Array, Float

from neuralab import nl


class ForecastModel(nnx.Module):
    def __init__(self, forecast_features: int, forecast_length: int, **kwargs):
        super().__init__(**kwargs)

        self.forecast_features = forecast_features
        self.forecast_length = forecast_length

        self.hippo_decoder = nl.HiPPODecoder(forecast_features, forecast_length)

    def forecast_loss(self, x, y, indices): ...


class H0(ForecastModel):
    def __init__(
        self,
        hippos: int = 4,
        emas: int = 4,
        in_channels: int = len(["price", "qty"]),
        forecast_features: int = 32,
        forecast_length: int = 512,
        hippo_features: int = 32,
        forecast: int = 512,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            forecast_features=forecast_features, 
            forecast_length=forecast_length
        )

        # input funnel
        self.hippo_enc = nl.HiPPOEncoder(4, hippo_features, rngs=rngs)
        self.ema = nl.EMA(4, rngs=rngs)

        in_features = hippos * in_channels * hippo_features + emas * in_channels

        self.in_norm = nnx.BatchNorm(in_features, rngs=rngs)

        # self.in_proj = nnx.Linear(self.in_features, features, rngs=rngs)
        # feed forward
        # self.feed = nl.FeedForward(features, rngs=rngs)

        self.out_proj = nnx.Linear(in_features, hippo_features, rngs=rngs)

        # output decoder
        self.hippo_decoder = nl.HiPPODecoder(hippo_features, forecast)

    def __call__(self, x: Float[Array, "L (p v) "]):

        std_x: Float[Array, "L (p v) emas"] = self.ema.standarize(x)
        hippo_x = self.hippo_enc(x)

        feat = np.concat(
            (
                rearrange(hippo_x, "l c h f -> l (c h f)"),
                rearrange(std_x, "l c e -> l (c e)"),
            ),
            axis=1,
        )

        feat = self.out_proj(feat)

        return feat  # np.concat([dc, feat], axis=-1)


# %%
if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from dataproxy.dataset import DATASETS

    rngs = nnx.Rngs(0)
    h0 = H0(rngs=rngs)

    DSI = 0
    x = np.log(
        np.stack(
            [
                DATASETS[DSI]["vwap"].to_numpy(),
                DATASETS[DSI]["vol"].to_numpy(),
            ],
            axis=1,
        )
    )

    out = h0(x[:1000])

    plt.pcolormesh(out)
    plt.colorbar()

# %%


class Trainer(nnx.Module):
    def __init__(
        self,
        model: H0,
        opt: nnx.Optimizer,
        *,
        forecast_features: int = 32,
        forecast_length: int = 512,
        rngs: nnx.Rngs,
    ):
        self.forecast_features = forecast_features
        self.forecast_length = forecast_length

        self.rngs = rngs
        self.model = model
        self.opt = opt

    @nnx.jit
    def step(self, x):
        def loss_fn(model):
            y = model(x)

            # reconstruction loss

            if isinstance(model, ForecastModel):
                # forecast_loss = model.forecast_loss(x, y, indices)
                ...

            indices = random.randint(
                self.rngs.next(), (1,), minval=512, maxval=len(x) - 512
            )

            x_i_to_n = lax.dynamic_slice(x[:, 0], indices, [512])

            indices = np.concat(
                [indices, np.zeros((1,), dtype=np.int32)],
                axis=0,
            )

            y_i = lax.dynamic_slice(y, indices, [1, 32])

            forecast = model.hippo_decoder(y_i)
            forecast_loss = np.mean(np.square(forecast - x_i_to_n))

            return forecast_loss

        loss, grad = nnx.value_and_grad(loss_fn)(self.model)
        self.opt.update(grad)
        return loss


if __name__ == "__main__":
    from dataproxy.dataset import DATASETS

    rngs = nnx.Rngs(0)
    h0 = H0(rngs=rngs)
    opt = nnx.Optimizer(h0, optax.adam(1e-3))
    trainer = Trainer(h0, opt, rngs=rngs)

    DSI = 0
    x = np.log(
        np.stack(
            [
                DATASETS[DSI]["vwap"].to_numpy(),
                DATASETS[DSI]["vol"].to_numpy(),
            ],
            axis=1,
        )
    )

    # nnx.pop(h0, State)
    for step in range(1000):
        loss = trainer.step(x)
        print(loss)
#%%
    from matplotlib import pyplot as plt

    y = h0(x)
    I = 5000
    forecast = h0.hippo_decoder(y[I])
    forecast_loss = np.mean(np.square(forecast - x[I : I + 512, 0]))
    plt.plot(x[I : I + 512, 0])
    #plt.plot(h0.hippo_decoder(y[I]))
    plt.show()
    print(forecast_loss)



