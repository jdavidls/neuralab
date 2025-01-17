from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal, Self

from jax import numpy as jnp, tree
from jaxtyping import Array, Float
from flax import struct
from matplotlib.pylab import f
from optax import losses

from neuralab import nl, settings
from neuralab.trading.dataset import Dataset
from neuralab.trading.evaluation import evaluate
from neuralab.trading.ground_truth import GroundTruth


@struct.dataclass()
class TrainDataset(nl.BaseDataset):
    dataset: Dataset
    labels: GroundTruth.Labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, *args) -> Self:
        return tree.map(lambda v: v.__getitem__(*args), self)


# type Batch = tuple[Dataset, GroundTruth.Labels]


class TradingModel(nl.Model):

    class Trainer(nl.Model.Trainer[TrainDataset, Dataset]):

        @dataclass()
        class Settings(nl.Model.Trainer.Settings):
            ground_truth: GroundTruth.Settings = field(
                default_factory=lambda: GroundTruth.Settings()
            )

            split_fraction: float = 0.75
            batch_size: int = 2048

        def prepare_session(self, dataset: Dataset):
            split_idx = int(len(dataset) * self.settings.split_fraction)
            training_set, evaluation_set = dataset[:split_idx], dataset[split_idx:]

            training_set = TrainDataset(
                training_set,
                GroundTruth.from_dataset(
                    training_set, self.settings.ground_truth
                ).labels,
            )
            return training_set.batched(
                self.settings.batch_size
            ), evaluation_set.batched(self.settings.batch_size)

        model: TradingModel
        settings: Settings # type: ignore

        def train_fn(self, batch: TrainDataset):

            self.model.settings.head_losses

            head_logits = self.model(batch.dataset)
            t, a, m, *head_shape = head_logits.shape

            assert (
                head_shape[0] == self.model.settings.num_heads
            ), f"Expected {self.model.settings.num_heads} heads, got {head_shape[0]}"

            def head_loss_fn(loss_fn: str, logits):
                match loss_fn:
                    case "mse":
                        return (
                            jnp.mean((logits - batch.labels.one_hot) ** 2, axis=-1)
                            * batch.labels.mask
                        )
                    case "cross_entropy":
                        return jnp.mean(
                            losses.softmax_cross_entropy(logits, batch.labels.one_hot)
                            * batch.labels.mask
                        )
                    case "max_total_perf":
                        return evaluate(logits, batch.dataset).loss()
                    case _:
                        raise ValueError(f"Unknown loss function {loss_fn}")

            loss_vals = [
                head_loss_fn(loss_fn, head_logits[..., i, :]) * loss_weight
                for i, (loss_fn, loss_weight) in enumerate(self.model.settings.head_losses.values())
            ]

            for key, loss in zip(self.model.settings.head_losses, loss_vals):
                setattr(self, f"{key}_loss", nl.Loss(loss))

        def eval_fn(self, dataset: Dataset):

            self.model.settings.head_losses

            head_logits = self.model(dataset)
            t, a, m, *head_shape = head_logits.shape

            assert (
                head_shape[0] == self.model.settings.num_heads
            ), f"Expected {self.model.settings.num_heads} heads, got {head_shape[0]}"



            for l, head in enumerate(self.model.settings.head_losses):
                logits = head_logits[..., l, :]
                evaluation = evaluate(logits, dataset)

                setattr(self, f"{head}_perf", nl.Metric(evaluation.total_performance))
                setattr(self, f"{head}_turn", nl.Metric(evaluation.total_turnover))


    @dataclass()
    class Settings(nl.Model.Settings):
        trainer: TradingModel.Trainer.Settings = field(
            default_factory=lambda: TradingModel.Trainer.Settings()
        )

        type LossFn = Literal["mse", "cross_entropy", "max_total_perf"]

        @property
        def num_heads(self):
            return len(self.head_losses)
        


        head_losses: dict[str, tuple[LossFn, float]] = field(
            default_factory=lambda: {
                "ce": ("cross_entropy", .5),
                "ev": ("max_total_perf", 1.),
            }
        )

    settings: Settings

    def __call__(self, x: Dataset) -> Float[Array, "logits"]:
        raise NotImplementedError
