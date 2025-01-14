# %%
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from flax import nnx, struct
from jax import numpy as jnp, lax
import optax
from neuralab import track
from neuralab.trading.dataset import Dataset


def calculate_risk_metrics(returns: jnp.ndarray, confidence_level: float = 0.95):
    """ Calculate VaR and max drawdown from log returns """
    var = -jnp.percentile(returns, (1 - confidence_level) * 100, axis=0)
    cumulative = jnp.cumsum(returns)
    running_max = lax.cummax(cumulative, axis=0)
    drawdown = running_max - cumulative
    max_drawdown = jnp.max(drawdown)
    return var, max_drawdown


@struct.dataclass
class Evaluation(struct.PyTreeNode):

    @dataclass(frozen=True)
    class Settings:

        # TODO: Los settings de los markets iran en el dataset provenientes
        # de knowloedge.
        # struct MarketSettings for stacking markets en realidad algunas de
        # estas settings de evaluacion son settings del dataframe cuando se
        # concatenan assets, los settings deben de concatenarse tambien por
        # ello estos settings deben ser structs cada mercado debe tener sus
        # propios settings. El dataset puede generar los settings
        # concatenados a traves de una propiedad

        @dataclass(frozen=True)
        class RiskControl:
            initial_margin: float = 0.1  # 10% initial margin
            maint_margin: float = 0.05  # 5% maintenance margin
            max_leverage: float = 5.0  # 5x maximum leverage
            var_limit: float = 0.02  # 2% VaR limit
            max_drawdown_limit: float = 0.15  # 15% max drawdown limit
            confidence_level: float = 0.95  # 95% confidence level

        transaction_cost: float = 0.001
        max_leverage: Optional[float] = 5.0
        risk_ctrl: Optional[RiskControl] = None

    dataset: Dataset
    settings: Settings = struct.field(pytree_node=False)
    turnover: jnp.ndarray
    returns: jnp.ndarray

    @property
    def transaction_cost(self):
        return jnp.log1p(self.turnover * self.settings.transaction_cost)

    @property
    def total_turnover(self):
        return jnp.sum(self.turnover, axis=0)

    @property
    def total_returns(self):
        return jnp.sum(self.returns, axis=0)

    @property
    def total_transaction_cost(self):
        return jnp.sum(self.transaction_cost, axis=0)

    @property
    def total_performance(self):
        return self.total_returns - self.total_transaction_cost

    @property
    def cum_returns(self):
        return jnp.cumsum(self.returns, axis=0)

    @property
    def cum_transaction_cost(self):
        return jnp.cumsum(self.transaction_cost, axis=0)

    @property
    def cum_performance(self):
        return self.cum_returns - self.cum_transaction_cost

    type LossMode = Literal["max_total_perf", "max_cum_perf"]

    def loss(self, mode: LossMode = "max_total_perf"):
        match mode:
            # case "max_mean_perf":
            #     return -jnp.mean(self.mean_performance)
            case "max_total_perf":
                return -jnp.mean(self.total_performance)
            case "max_cum_perf":
                return -jnp.mean(self.cum_performance)

def evaluate(
    logits: jnp.ndarray,
    self: Dataset,
    *,
    settings: Evaluation.Settings = Evaluation.Settings(),
) -> Evaluation:
    """Simulate single asset portfolio"""
    t, a, m, *head_shape = logits.shape

    match head_shape:
        case ():
            probs = nnx.hard_sigmoid(logits)
            turnover = jnp.abs(jnp.diff(probs, append=logits[-1:], axis=0))
            returns = probs * self.returns

        case (3,):
            probs = nnx.softmax(logits)

            probs_out = probs[:, :, :, 1]
            probs_long = probs[:, :, :, 0]
            probs_short = probs[:, :, :, 2]

            turnover = jnp.abs(jnp.diff(probs_long, axis=0)) + jnp.abs(
                jnp.diff(probs_short, axis=0)
            )
            returns = probs_long * self.returns - probs_short * self.returns

        case (4,):
            leverage, logits = logits[..., 0], logits[..., 1:4]
            leverage = nnx.sigmoid(leverage) * settings.max_leverage

            probs = nnx.softmax(logits)

            probs_out = probs[:, :, :, 1]
            probs_long = probs[:, :, :, 0] * leverage
            probs_short = probs[:, :, :, 2] * leverage

            turnover = jnp.abs(jnp.diff(probs_long, axis=0)) + jnp.abs(
                jnp.diff(probs_short, axis=0)
            )
            returns = probs_long * self.returns - probs_short * self.returns
        case _:
            raise ValueError(f"Invalid logits shape: {logits.shape}")

    return Evaluation(
        settings=settings,
        dataset=self,
        turnover=turnover,
        returns=returns,
    )


def fit_target_logits(
    dataset: Dataset,
    num_epochs: int = 10000,
    lr: float = 1e-2,
    # loss_mode: Evaluation.LossMode = "max_total_perf",
):
    from neuralab.trading.evaluation import evaluate

    with track.task(description="Fitting dataset", total=num_epochs) as task:
        target_logits = dataset.returns
        opt = optax.nadam(lr)
        opt_state = opt.init(target_logits)

        try:

            @nnx.jit
            def fit_step(opt_state, dataset, logits):

                def loss_fn(logits):
                    return evaluate(logits, dataset).loss()

                loss, grads = nnx.value_and_grad(loss_fn)(logits)
                updates, opt_state = opt.update(grads, opt_state)
                logits = optax.apply_updates(logits, updates)

                return opt_state, logits, loss, jnp.std(grads)

            for n in range(num_epochs):
                opt_state, target_logits, loss, g_std = fit_step(opt_state, dataset, target_logits)

                if n % 25 == 0:
                    # EVALUATION SETTINGS debe saber que activacion aplicar a los logits hard_sigmoid, sign relu etc..
                    eval = evaluate(nnx.relu(jnp.sign(target_logits)), dataset)  # type: ignore
                    perf = jnp.mean(eval.total_performance)
                    cost = jnp.mean(eval.total_transaction_cost)
                    turnover = jnp.mean(eval.total_turnover)
                    w_std = jnp.std(target_logits)  # type: ignore

                task.update(
                    completed=n,
                    description=(
                        f"Gain: {-loss:.2f} "
                        f"Perf: {perf:.2f} "  # type: ignore
                        f"Cost: {cost:.2f} "  # type: ignore
                        f"Turnover: {turnover:.0f} "  # type: ignore
                        f"Weights: ±{w_std:.2f}"  # type: ignore
                    ),
                )
        except KeyboardInterrupt:
            pass

        return target_logits
