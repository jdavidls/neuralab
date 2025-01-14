from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

from einops import rearrange
from jax import numpy as jnp, tree
from jaxtyping import Array, Float
from optax import losses

from neuralab import nl
from neuralab.trading.dataset import Dataset
from neuralab.trading.evaluation import evaluate
from neuralab.trading.ground_truth import GroundTruth


class TradingModel(nl.Model):

    @dataclass(frozen=True)
    class Settings(nl.Model.Settings):
        training: TradingModel.Training.Settings = field(
            default_factory=lambda: TradingModel.Training.Settings()
        )

    class Training(nl.Model.Training):

        @dataclass(frozen=True)
        class Settings(nl.Model.Training.Settings):
            ground_truth: GroundTruth.Settings = field(
                default_factory=lambda: GroundTruth.Settings()
            )
            # batch_size = 4096
            loss_fn: Literal["mse", "cross_entropy", "max_total_perf"] = "cross_entropy"

        type Batch = tuple[Dataset, GroundTruth.Labels]

        class Session(nl.Model.Training.Session):

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
            logits = self.model(x)

            match self.settings.loss_fn:
                case "mse":
                    loss = jnp.mean((logits - y.one_hot) ** 2, axis=-1) * y.mask

                case "cross_entropy":
                    loss = jnp.mean(
                        losses.softmax_cross_entropy(logits, y.one_hot) * y.mask
                    )

                case "max_total_perf":
                    loss = evaluate(logits, x).loss()

            return loss

        def __init__(self, model: TradingModel):
            super().__init__(model)

    @abstractmethod
    def __call__(self, x: Dataset) -> Float[Array, "logits"]: ...
