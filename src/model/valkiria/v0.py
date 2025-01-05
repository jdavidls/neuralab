# %%
# VAE doble donde se entrenan dos autoencoders en una ventana de tiempo
# partida por la mitad (el sample de referencia que representa el momento presente)

from jax import checking_leaks, numpy as jnp
from flax import nnx
import optax

from neuralab.fn import cosine_distance, random_slice
from neuralab.fn.distance import euclidean_squared_distance
from neuralab.nl.hippo import HiPPODecoder, HiPPOEncoder
from neuralab.nl.vae import VAE


class V0(nnx.Module):

    def __init__(
        self,
        features: list[int] = [64, 32, 16],
        length: int = 512,
        *,
        rngs: nnx.Rngs,
    ):
        self.features = features
        self.length = length
        self.rngs = rngs

        self.hippo_enc = HiPPOEncoder(features[0], dt=1 / length)
        self.hippo_dec = HiPPODecoder(features[0], length, dt=1 / length)

        self.hippo_norm = nnx.BatchNorm(features[0], use_bias=False, rngs=rngs)

        self.left_vae = VAE(features, rngs=rngs)
        self.right_vae = VAE(features, rngs=rngs)

    def split_window(self, x):
        left = x[: self.length]
        right = x[self.length - 1 :: -1]

        return left - left[-1], right - right[-1]  # zero-centered

    def __call__(self, x):
        l, r = self.split_window(x)
        l_hippo = self.hippo_enc(l)[-1]
        r_hippo = self.hippo_enc(r)[-1]

        #l_hippo = self.hippo_norm(l_hippo)
        #r_hippo = self.hippo_norm(r_hippo)

        l_vae = self.left_vae(l_hippo)
        r_vae = self.right_vae(r_hippo)

        return l_vae, r_vae, l_hippo, r_hippo

    def loss(self, x):
        l_vae, r_vae, l_hippo, r_hippo = self(x)
        l_recons, l_mu, l_logvar = l_vae
        r_recons, r_mu, r_logvar = r_vae

        latent_distance = euclidean_squared_distance(l_mu, r_mu)

        loss = (
            +VAE.se_loss(l_hippo, l_recons)
            + VAE.kl_loss(l_mu, l_logvar)
            + VAE.se_loss(r_hippo, r_recons)
            + VAE.kl_loss(r_mu, r_logvar)
            + latent_distance
        )

        return loss, {
            "loss": loss,
            "latent_distance": latent_distance,
        }


class Train(nnx.Module):
    def __init__(
        self,
        model: V0,
        opt: nnx.Optimizer,
        *,
        batch_size=32,
        rngs: nnx.Rngs,
    ):
        self.model = model
        self.opt = opt
        self.rngs = rngs
        self.batch_size = batch_size


@nnx.jit
def train_step(model, opt, batch):

    def loss_fn(model):
        loss, aux = model.loss(batch)

        return jnp.mean(loss), {k: jnp.mean(v) for k, v in aux.items()}

    (_, aux), grad = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    opt.update(grad)
    return aux


# %%
if __name__ == "__main__":
    from neuralab.ds.dataset import DATASETS

    rngs = nnx.Rngs(0)
    model = V0(rngs=rngs)
    opt = nnx.Optimizer(model, optax.adam(1e-3))
    # train = Train(model, opt, rngs=rngs)

    # %%
    model.train()
    # with checking_leaks():
    for n in range(1024):
        dataset = DATASETS[n % len(DATASETS)]["vwap"].to_numpy()
        batch, _ = random_slice(
            dataset,
            model.length * 2 - 1,
            32,
            out_axis=1,
            rngs=rngs,
        )

        aux = train_step(model, opt, jnp.log(batch))
        print(f"{n} {aux}")

    # %%
    from matplotlib import pyplot as plt

    dataset = DATASETS[0]["vwap"].to_numpy()

    x = random_slice(
        dataset,
        model.length * 2 - 1,
        1,
        rngs=rngs,
    )[0][0]
    
    model.eval()

    l, r = model.split_window(jnp.log(x))

    l_vae, r_vae, l_hippo, r_hippo = model(x)

    l_recons, l_mu, l_logvar = l_vae
    r_recons, r_mu, r_logvar = r_vae

    # %%

    plt.plot(l_hippo)
    # plt.plot(l_recons)
