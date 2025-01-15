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
from typing import ClassVar, Generator, Optional

import optax
from flax import nnx, struct, typing as flax_typing
from jax import numpy as jnp
from jax import tree
from jaxtyping import Array, Float
from zmq import has

from neuralab import track
from neuralab.nl.common import Loss, Metric
from neuralab.resource import Resource


class StopTraining(BaseException): ...


class LossNonFiniteException(ValueError): ...


class Model(nnx.Module):
    """
    Un modelo es un modulo entrenable: define un trainer.
    """

    @dataclass(frozen=True)
    class Ref(Resource.Ref):
        ...#nnx.split

    @dataclass()
    class Settings(ABC):
        training: Model.Training.Settings

    class Training(nnx.Module):  # a abstract class of trainer
        """
        El trainer puede generar datos agregados al dataset (hippo features)
        tambien puede re-asignar parametros de forma dinamica
        trainer accede a model pero no tiene model dentro de el...
        """

        @dataclass()
        class Settings(ABC):
            optimizer: optax.GradientTransformation = optax.adam(1e-3)

        class Session:
            def __init__(self, training: Model.Training):
                """
                Initialize the session with the training object.
                """
                self.training = training

            @abstractmethod
            def prepare_dataset(self, *args, **kwargs):
                """
                Prepare the session with the given arguments.
                """
                raise NotImplementedError

            @abstractmethod
            def __len__(self) -> int:
                """
                Return the length of the session.
                """
                pass

            @abstractmethod
            def __iter__(self):
                """
                Iterate over the session.
                """
                yield ...

            def dispose_dataset(self):
                """
                Dispose of the session resources.
                """
                pass

            @contextmanager
            def enter_dataset(self, *args, **kwargs):
                """
                Context manager to enter a dataset.
                """
                try:
                    self.prepare_dataset(*args, **kwargs)
                    yield self
                finally:
                    self.dispose_dataset()

            @contextmanager
            def enter_epoch(self):
                """
                Context manager for an epoch.
                """
                yield self

            def prepare_batch(self, batch):
                """
                Prepare a batch for training.
                """
                return batch

            def update_params(self, grad):
                """
                Update the parameters using the optimizer.
                """
                self.training.optimizer.update(grad)

        def __init__(self, model: Model):
            self.settings = model.settings.training
            self.model = model
            self.optimizer = nnx.Optimizer(self, self.settings.optimizer)

        session: Session
        main_loss: Loss

        @contextmanager
        def enter_session(self):
            try:
                self.session = self.Session(self)
                yield self.session
            finally:
                del self.session

        def __call__(self, batch):
            """
            Train step function.
            """
            self.loss(batch)

            metrics = nnx.pop(self, Metric).flat_state()

            loss = sum(
                jnp.mean(loss.value)
                for loss in metrics.values()
                if issubclass(loss.type, Loss)
            )

            metrics = {_metric_path_to_str(k): v.value for k, v in metrics.items()}

            return loss, metrics

        @abstractmethod
        def loss(self, batch) -> Float[Array, "..."]: ...

    @cached_property
    def training(self):
        return self.Training(self)  # type: ignore

    def __init__(self, ref: Ref, settings: Settings):
        self.ref = ref
        self.settings = settings


def _metric_path_to_str(path: flax_typing.PathParts):
    return ".".join(str(part) for part in path)


## puede almacenarse la estructura fit entera para suspender un entrenamiento
## fit geenra un identificador unico y almacena los datos de entrenamiento.
class Fit[M: Model]:
    """
    Fit se encarga de gestionar el proceso de entrenamiento de un modelo

    model_fit = fit(model) transforma el modelo en un objeto entrenable

    model_fit(dataset) entrena el modelo (en un hilo separado)
    """

    def __init__(self, model: M):
        self.trainer = self.Trainer(model)

    class Trainer:
        def __init__(
            self,
            model: Model,
            *,
            evaluate_every=10,
            try_batch_scan=False,
        ):
            self.training = model.training
            self.evaluate_every = evaluate_every
            self.try_batch_scan = try_batch_scan
            self.history: Optional[dict[str, Float[Array, "losses and metrics"]]] = None

        current_session: Optional[Model.Training.Session] = None

        def __call__(self, *args, **kwargs):
            return self.run_session(*args, **kwargs)

        def _append_history(self, metrics):
            if self.history is not None:
                for k, v in metrics.items():
                    self.history[k] = jnp.append(self.history[k], v)
            else:
                self.history = metrics

        def run_session(self, *args, num_epochs: int, **kwargs):
            try:

                with (
                    track.task(
                        total=num_epochs,
                        description="[EPOCH] Preparing...",
                    ) as trackbar,
                    self.training.enter_session() as session,
                    session.enter_dataset(*args, **kwargs),
                ):

                    trackbar.update(
                        description=(f"[EPOCH {1}/{num_epochs}] Running..."),
                    )

                    for epoch in range(num_epochs):
                        train_metrics = self.run_epoch(session)

                        self._append_history(train_metrics)

                        trackbar.update(
                            description=(
                                f"[EPOCH {epoch+1}/{num_epochs}] {_dump_metrics(train_metrics)}"
                            ),
                            advance=1,
                        )

                        if epoch % self.evaluate_every == 0:
                            self.run_evaluation(session)

            except KeyboardInterrupt:
                pass

        def run_epoch(self, session):
            # Batch scan
            # if batch_scan:
            #     @nnx.scan(in_axes=0, out_axes=0)
            #     def batch_scanner(xx, yy):
            #         return self.train_step(xx, yy)
            #     return batch_scanner(x, y)

            with (
                session.enter_epoch() as epoch,
                track.task(
                    total=len(epoch),
                    description="[BATCH] Preparing...",
                ) as trackbar,
            ):

                trackbar.update(
                    description=(f"[BATCH {1}/{len(epoch)}] Running..."),
                )

                epoch_metrics = []

                for n, batch in enumerate(epoch):
                    metrics = _run_batch(self.training, batch)

                    trackbar.update(
                        description=(
                            f"[BATCH {n+1}/{len(epoch)}] {_dump_metrics(metrics)}"
                        ),
                        advance=1,
                    )

                    if not jnp.isfinite(metrics["loss"]):
                        raise LossNonFiniteException(f"Loss is {metrics["loss"]}")

                    epoch_metrics.append(metrics)

                return tree.map(_reduce_epoch_metrics, *epoch_metrics)

        def run_evaluation(self, session):
            pass

    def __call__(self, *args, **kwargs):
        return self.trainer.run_session(*args, **kwargs)

    thread: Thread

    def start(self, *args, **kwargs):
        """
        Start the training process.
        """
        if hasattr(self, "thread"):
            raise RuntimeError("Training is already running.")
        self.thread = Thread(target=self, args=args, kwargs=kwargs)
        self.thread.start()
        return self.thread

    def stop(self):
        """
        Stop the training process.
        """
        if not hasattr(self, "thread"):
            raise RuntimeError("Training is not running.")
        # signals to stop th thread
        self.thread.join()
        del self.thread

    @property
    def history(self):
        return self.trainer.history


def _dump_metrics(metrics: dict[str, jnp.ndarray]):
    metrics = tree.map(jnp.mean, metrics)
    return " ".join(f"{k}: {v:.4f}" for k, v in metrics.items())


def fit[M: Model](model: M) -> Fit[M]:
    """
    Transforma el modelo es un objeto entrenable
    """
    return Fit(model)


@nnx.jit
def _run_batch(
    training: Model.Training,
    batch,
) -> dict[str, jnp.ndarray]:

    # batch = training.session.prepare_batch(batch)

    def loss_fn(training):
        loss, metrics = training(batch)
        metrics["loss"] = loss
        loss = jnp.nan_to_num(loss, nan=0, posinf=0, neginf=0)
        return loss, metrics

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (_, metrics), grad = grad_fn(training)

    training.optimizer.update(grad)

    return metrics


def _reduce_epoch_metrics(*values: list[jnp.ndarray]) -> jnp.ndarray:
    return jnp.mean(jnp.array(values))[None]


# %%
