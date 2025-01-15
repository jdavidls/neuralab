from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

from einops import rearrange
from jax import numpy as jnp, tree
from jaxtyping import Array, Float
from matplotlib.pylab import f
from optax import losses

from neuralab import nl
from neuralab.trading.dataset import Dataset
from neuralab.trading.evaluation import evaluate
from neuralab.trading.ground_truth import GroundTruth


class TradingModel(nl.Model):

    @dataclass()
    class Settings(nl.Model.Settings):
        training: TradingModel.Training.Settings = field(
            default_factory=lambda: TradingModel.Training.Settings()
        )

    class Training(nl.Model.Training):
        model: TradingModel

        @dataclass()
        class Settings(nl.Model.Training.Settings):
            ground_truth: GroundTruth.Settings = field(
                default_factory=lambda: GroundTruth.Settings()
            )

            type LossFn = Literal["mse", "cross_entropy", "max_total_perf"]

            @property
            def num_heads(self):
                return len(self.head_losses)

            head_losses: dict[str, LossFn] = field(
                default_factory=lambda: {
                    "ce": "cross_entropy",
                    "ev": "max_total_perf",
                }
            )

        type Batch = tuple[Dataset, GroundTruth.Labels]

        class Session(nl.Model.Training.Session):
            training: TradingModel.Training

            def prepare_dataset(self, dataset: Dataset, batch_size: int = 2048):
                self.dataset = dataset
                self.batch_size = batch_size
                self.ground_truth = GroundTruth.from_dataset(
                    dataset, self.training.settings.ground_truth
                )

                # x = dataset
                # y = ground_truth.labels

                # batch_length = 4096
                # batch_count = x.shape[0] // batch_length
                # rem = x.shape[0] % batch_length

                # if rem:
                #     print("Removing rem")
                #     x = x[:-rem]
                #     y = y[:-rem]

                # def batch_rearrange(x):
                #     return rearrange(
                #         x, "t (bl bc) ... -> bc t bl ...", bl=batch_length, bc=batch_count
                #     )

                # self.x = tree.map(batch_rearrange, x)
                # self.y = tree.map(batch_rearrange, y)

            def __len__(self):
                return len(self.dataset) // self.batch_size

            def __iter__(self):  # training_set
                for n, m in pairwise(range(0, len(self.dataset) + 1, self.batch_size)):
                    yield self.dataset[n:m], self.ground_truth.labels[n:m]

                # batch rearrange
                # if batch_size is not None:

            def dispose_dataset(self):
                del self.dataset
                del self.ground_truth

        settings: Settings
        model: TradingModel

        def loss(self, batch: Batch):
            x, y = batch
            head_logits = self.model(x)
            t, a, m, *head_shape = head_logits.shape

            assert (
                head_shape[0] == self.settings.num_heads
            ), f"Expected {self.settings.num_heads} heads, got {head_shape[0]}"

            def head_loss_fn(fn, logits):
                match fn:
                    case "mse":
                        return jnp.mean((logits - y.one_hot) ** 2, axis=-1) * y.mask
                    case "cross_entropy":
                        return jnp.mean(
                            losses.softmax_cross_entropy(logits, y.one_hot) * y.mask
                        )
                    case "max_total_perf":
                        return evaluate(logits, x).loss()
                    case _:
                        raise ValueError(f"Unknown loss function {fn}")

            loss_vals = [
                head_loss_fn(fn, head_logits[..., i, :])
                for i, fn in enumerate(self.settings.head_losses.values())
            ]

            for key, loss in zip(self.settings.head_losses, loss_vals):
                setattr(self, f"{key}_loss", nl.Loss(loss))

        def __init__(self, model: TradingModel):
            super().__init__(model)

    def __call__(self, x: Dataset) -> Float[Array, "logits"]:
        raise NotImplementedError
