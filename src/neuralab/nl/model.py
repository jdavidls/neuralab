# %%
"""
Establece un entorno visual en el que renderizar el transcurso del entrenamiento..

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from threading import Lock, Thread
from typing import (
    Iterator,
    Optional,
    Protocol,
    Self,
    overload,
)

from attrs import field
import optax
from flax import nnx, struct, typing as flax_typing
from jax import numpy as jnp
from jax import tree
from jaxtyping import Array, Float

from neuralab import track
from neuralab.nl.common import Loss, Metric, reset_state

from itertools import pairwise
from typing import Generator, Sequence, overload
from flax import struct


class StopTraining(BaseException): ...


class LossNonFiniteException(ValueError): ...


class BaseDataset(struct.PyTreeNode):

    @property
    def shape(self) -> list[int]:
        leaves = tree.leaves(self)

        if len(leaves) == 0:
            return []

        assert all(a.shape == b.shape for a, b in zip(leaves[:-1], leaves[1:]))

        return leaves[0].shape

    def __len__(self) -> int:
        return self.shape[0] if len(self.shape) > 0 else 0

    def __getitem__(self, *args) -> Self:
        return tree.map(lambda v: v.__getitem__(*args), self)

    def batched(self, batch_size: int) -> BatchedDataset[Self]:
        return BatchedDataset(self, batch_size)


class DatasetLike(Protocol):

    @property
    def shape(self) -> list[int]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, *args) -> Self: ...


class BatchedDataset[T: DatasetLike]:
    __slots__ = ("dataset", "batch_size")

    def __init__(self, dataset: T, batch_size: int):
        self.batch_size = batch_size
        self.dataset = dataset

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    @overload
    def __getitem__(self, index: int) -> T: ...

    def __getitem__(self, index):  # -> T | Self:

        if isinstance(index, slice):
            return BatchedDataset(
                self.dataset[
                    index.start * self.batch_size : index.stop * self.batch_size
                ],
                self.batch_size * index.step,
            )

        if index < 0:
            index += len(self)

        return self.dataset[index * self.batch_size : (index + 1) * self.batch_size]

    def __iter__(self) -> Iterator[T]:
        for n, m in pairwise(range(0, len(self.dataset) + 1, self.batch_size)):
            yield self.dataset[n:m]


class Model(nnx.Module):
    """
    Un modelo es un modulo entrenable: define un trainer.
    """

    class Trainer[TrainBatch: DatasetLike, EvalBatch: DatasetLike](nnx.Module):

        @dataclass()
        class Settings:
            optimizer: optax.GradientTransformation = optax.adam(1e-3)

        settings: Settings

        def __init__(self, model: Model, settings: Settings):
            self.model = model
            self.settings = settings
            self.optimizer = nnx.Optimizer(self, self.settings.optimizer)

        @contextmanager
        def session(self, *args, **kwargs):
            yield self.prepare_session(*args, **kwargs)

        @abstractmethod
        def prepare_session(self, *args, **kwargs) -> tuple[BatchedDataset[TrainBatch], BatchedDataset[EvalBatch]]: ...

        @abstractmethod
        def train_fn(self, batch: TrainBatch): ...

        @abstractmethod
        def eval_fn(self, batch: EvalBatch): ...

    @dataclass()
    class Settings:
        trainer: Model.Trainer.Settings

    def __init__(self, settings: Settings):
        self.settings = settings



## puede almacenarse la estructura fit entera para suspender un entrenamiento
## fit geenra un identificador unico y almacena los datos de entrenamiento.
class Training[T: Model.Trainer]:
    """
    Fit se encarga de gestionar el proceso de entrenamiento de un modelo

    model_fit = fit(model) transforma el modelo en un objeto entrenable

    model_fit(dataset) entrena el modelo (en un hilo separado)
    """

    def __init__(self, trainer: T):
        self.trainer = trainer  # type: ignore
        self.train_history: Optional[dict[str, Float[Array, "losses and metrics"]]] = (
            None
        )
        self.eval_history: Optional[dict[str, Float[Array, "losses and metrics"]]] = (
            None
        )
        self.thread = None

    def __call__(self, *args, **kwargs):
        """
        Start the training process.
        """
        if self.thread is not None:
            raise RuntimeError("Training is already running.")
        self.thread = Thread(target=self.run_session, args=args, kwargs=kwargs)
        self.thread.start()

    def run_session(self, *args, num_epochs: int, **kwargs):
        try:
            with (
                track.task(
                    total=num_epochs,
                    description="[EPOCH] Preparing...",
                ) as trackbar,
                self.trainer.session(*args, **kwargs) as (training_set, evaluation_set),
            ):

                trackbar.update(
                    description=(f"[EPOCH {1}/{num_epochs}] Running..."),
                )

                for epoch in range(num_epochs):
                    train_metrics = self.run_training(training_set)
                    eval_metrics = self.run_evaluation(evaluation_set)

                    trackbar.update(
                        description=(
                            f"[EPOCH {epoch+1}/{num_epochs}] {_dump_metrics(train_metrics)} {_dump_metrics(eval_metrics)}"
                        ),
                        advance=1,
                    )

        except KeyboardInterrupt:
            pass

    def run_training(self, training_set: BatchedDataset):
        # Batch scan
        # if batch_scan:
        #     @nnx.scan(in_axes=0, out_axes=0)
        #     def batch_scanner(xx, yy):
        #         return self.train_step(xx, yy)
        #     return batch_scanner(x, y)

        with track.task(description="[TRAIN] Preparing...") as trackbar:

            self.trainer.train(training=True)
            reset_state(self.trainer)

            train_metrics = []

            trackbar.update(
                total=len(training_set),
                description=(f"[TRAIN {0}/{len(training_set)}]"),
            )

            for n, batch in enumerate(training_set):
                metrics = _run_training_batch(self.trainer, batch)

                trackbar.update(
                    description=(
                        f"[TRAIN {n+1}/{len(training_set)}] {_dump_metrics(metrics)}"
                    ),
                    advance=1,
                )

                if not jnp.isfinite(metrics["loss"]):
                    raise LossNonFiniteException(f"Loss is {metrics["loss"]}")

                train_metrics.append(metrics)


            train_metrics = tree.map(_reduce_epoch_metrics, *train_metrics)

            if self.train_history is not None:
                for k, v in train_metrics.items():
                    self.train_history[k] = jnp.append(self.train_history[k], v, axis=0)
            else:
                self.train_history = train_metrics

            return train_metrics

    def run_evaluation(self, evaluation_set: BatchedDataset):
        with track.task(description="[EVAL] Preparing...") as trackbar:
            self.trainer.eval(training=False)
            reset_state(self.trainer)

            evaluation_metrics = []

            trackbar.update(
                total=len(evaluation_set),
                description=(f"[EVAL {0}/{len(evaluation_set)}]"),
            )

            for n, batch in enumerate(evaluation_set):
                metrics = _run_evaluation_batch(self.trainer, batch)

                trackbar.update(
                    description=(
                        f"[EVAL {n+1}/{len(evaluation_set)}] {_dump_metrics(metrics)}"
                    ),
                    advance=1,
                )

                evaluation_metrics.append(metrics)

            evaluation_metrics = tree.map(_reduce_epoch_metrics, *evaluation_metrics)

            if self.eval_history is not None:
                for k, v in evaluation_metrics.items():
                    self.eval_history[k] = jnp.append(self.eval_history[k], v, axis=0)
            else:
                self.eval_history = evaluation_metrics

            return evaluation_metrics

    def stop(self):
        """
        Stop the training process.
        """
        if self.thread is None:
            raise RuntimeError("Training is not running.")
        # signals to stop th thread
        # self.
        self.thread.join()
        self.thread = None

    

def _metric_path_to_str(path: flax_typing.PathParts):
    return ".".join(str(part) for part in path)


def _reduce_epoch_metrics(*values: list[jnp.ndarray]) -> jnp.ndarray:
    return jnp.mean(jnp.array(values))[None, ...]

def _dump_metrics(metrics: dict[str, jnp.ndarray]):
    metrics = tree.map(jnp.mean, metrics)
    return " ".join(f"{k}: {v:.4f}" for k, v in metrics.items())


def fit(model: Model) -> Training:
    """
    Transforma el modelo es un objeto entrenable
    """
    return Training(model.Trainer(model, model.settings.trainer)) # type: ignore


@nnx.jit
def _run_training_batch(
    trainer: Model.Trainer,
    batch,
) -> dict[str, jnp.ndarray]:

    def train(trainer):
        trainer.train_fn(batch)

        metrics = nnx.pop(trainer, Metric).flat_state()

        loss = sum(
            jnp.mean(loss.value)
            for loss in metrics.values()
            if issubclass(loss.type, Loss)
        )

        metrics = {_metric_path_to_str(k): v.value for k, v in metrics.items()}
        metrics = {"loss": loss, **metrics}

        loss = jnp.nan_to_num(loss, nan=0, posinf=0, neginf=0)
        return loss, metrics

    grad_fn = nnx.value_and_grad(train, has_aux=True)
    (_, metrics), grad = grad_fn(trainer)

    trainer.optimizer.update(grad)

    return metrics


@nnx.jit
def _run_evaluation_batch(
    trainer: Model.Trainer,
    batch,
) -> dict[str, jnp.ndarray]:
    # batch = training.session.prepare_batch(batch)

    trainer.eval_fn(batch)
    metrics = nnx.pop(trainer, Metric).flat_state()
    metrics = {_metric_path_to_str(k): v.value for k, v in metrics.items()}
    return metrics



# %%
