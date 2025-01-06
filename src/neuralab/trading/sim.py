# %%
"""
This module implements investment portfolio simulation using unsupervised learning.

The simulation functions take a tensor of log asset prices and parameters that
define the portfolio behavior. The simulation model is trained using an optimization
algorithm that minimizes the loss function, defined as the difference between
the expected return of the simulated portfolio and a reference portfolio.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Optional


from flax import nnx, struct
from jax import lax
from jax import numpy as jnp

from neuralab.trading.dataset import Dataset

@dataclass
class RiskControl:
    initial_margin: float = 0.1  # 10% initial margin
    maint_margin: float = 0.05  # 5% maintenance margin
    max_leverage: float = 5.0  # 5x maximum leverage
    var_limit: float = 0.02  # 2% VaR limit
    max_drawdown_limit: float = 0.15  # 15% max drawdown limit
    confidence_level: float = 0.95  # 95% confidence level


@dataclass
class SimParams:
    transaction_cost: float = 0.001
    max_leverage: Optional[float] = 5.0
    risk_ctrl: Optional[RiskControl] = None


@struct.dataclass
class SimMetrics(struct.PyTreeNode):
    dataset: Dataset
    params: SimParams = struct.field(pytree_node=False)
    turnover: jnp.ndarray
    returns: jnp.ndarray

    @property
    def transaction_cost(self):
        return jnp.log1p(self.turnover * self.params.transaction_cost)

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

    def loss(self, mode: str = "max_total_perf"):
        match mode:
            case "max_total_perf":
                return -jnp.mean(self.total_performance)
            case "max_cum_perf":
                return -jnp.mean(self.cum_performance)


def calculate_risk_metrics(returns: jnp.ndarray, confidence_level: float = 0.95):
    """Calculate VaR and max drawdown from log returns"""
    var = -jnp.percentile(returns, (1 - confidence_level) * 100, axis=0)
    cumulative = jnp.cumsum(returns)
    running_max = lax.cummax(cumulative, axis=0)
    drawdown = running_max - cumulative
    max_drawdown = jnp.max(drawdown)
    return var, max_drawdown


def sim(
    dataset: Dataset,
    weights: jnp.ndarray,  ## can be
    *,
    params: SimParams = SimParams(),
) -> SimMetrics:
    """Simulate single asset portfolio"""

    # if params.max_leverage is not None:
    # weights *= 1e1

    turnover = jnp.abs(jnp.diff(weights, append=weights[-1:], axis=0))

    # if params.portfolio_mode is True:
    # turnover = jnp.sum(turnover, axis=1, keepdims=True)

    # total_turnover = jnp.sum(turnover, axis=0)
    # log_costs = jnp.log1p(-total_turnover * params.transaction_cost)

    returns = jnp.clip(weights, 0, 1) * dataset.returns

    # print(log_costs.shape)

    if params.risk_ctrl is not None:
        ...

    return SimMetrics(
        dataset=dataset,
        params=params,
        turnover=turnover,
        returns=returns,
    )


# def sim_loss(
#     metrics: SimMetrics,
#     # fut_metrics: Optional[SimMetrics] = None,
#     target_returns: Optional[jnp.ndarray] = None,
# ):
#     """Calculate loss between simulated and target portfolio in log-space"""


#     if target_returns is None:
#         # Maximize sharpe ratio
#         base_loss = -jnp.mean(metrics.returns) / (jnp.std(metrics.returns) + 1e-6)
#     else:
#         # Minimize squared error
#         base_loss = jnp.mean((metrics.returns - target_returns) ** 2)

#     # risk_penalty = metrics.var + metrics.max_drawdown

#     return base_loss  # + risk_penalty


# %%
# trading.labels
import optax
from jax import random
from tqdm import trange


def fit_labels(
    dataset: Dataset,
    labels: Optional[jnp.ndarray] = None,
    epochs=1000,
    lr=1e-2,
    mode="max_total_perf",
):
    # from neuralab.nl.sim import sim, SimParams
    if labels is None:
        labels = dataset.returns

    opt = optax.nadam(lr)
    opt_state = opt.init(labels)
    sim_params = SimParams(transaction_cost=0.001)

    fit_activation = nnx.hard_sigmoid
    eval_activation = lambda x: nnx.relu(jnp.sign(x))

    @nnx.jit
    def fit_step(opt_state, labels, dataset):

        def loss_fn(labels):
            return sim(dataset, fit_activation(labels), params=sim_params).loss(mode)

        loss, grads = nnx.value_and_grad(loss_fn)(labels)
        updates, opt_state = opt.update(grads, opt_state)
        labels = optax.apply_updates(labels, updates)

        return opt_state, labels, loss, jnp.std(grads)

    try:
        tqdm = trange(epochs)
        for n in tqdm:
            if n % 25 == 0:
                s = sim(dataset, eval_activation(labels))
                perf = jnp.mean(s.total_performance)
                cost = jnp.mean(s.total_transaction_cost)
                turnover = jnp.mean(s.total_turnover)

            opt_state, labels, mode, g_std = fit_step(opt_state, labels, dataset)
            l_std = jnp.std(labels)

            tqdm.set_description(
                f"Gain: {-mode:.2f} "
                f"Perf: {perf:.2f} "
                f"Cost: {cost:.2f} "
                f"Turn: {turnover:.2f} "
                f"Labels: ±{l_std:.2f}"
            )

    except KeyboardInterrupt:
        pass

    return labels


# %%
if __name__ == "__main__":

    dataset = Dataset.load()
    rngs = nnx.Rngs(0)

    labels = fit_labels(dataset)

    # %%
    from matplotlib import pyplot as plt
    from einops import rearrange
    import numpy as np

    TIMESLICE = slice(0, 2500)
    ASSET = 0
    behaviour = nnx.relu(jnp.sign(labels))[TIMESLICE, ASSET].T
    price = dataset.log_price[TIMESLICE, ASSET]

    x = np.arange(len(price))
    y = np.linspace(np.min(price), np.max(price), len(behaviour))
    X, Y = np.meshgrid(x, y)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot behavior with correct scaling
    mesh = ax.pcolormesh(X, Y, behaviour, cmap="RdYlBu", vmin=0, vmax=1)

    # Plot price line on top
    ax.plot(x, price, label=["spot", "fut"])

    # Add colorbar and labels
    plt.colorbar(mesh, label="Behavior")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title("Price over Behavior Background")
    ax.legend()

    plt.tight_layout()
    plt.show()

    # %%
    from jaxtyping import Array, Float, Int
    from jax import lax, tree
    from functools import cached_property
    ## Events and Trends are Labels


    @struct.dataclass
    class Events(struct.PyTreeNode):
        dataset: Dataset
        asset: Int[Array, "..."]
        market: Int[Array, "..."]
        at: Int[Array, "..."]


    @struct.dataclass
    class Trends(struct.PyTreeNode):
        dataset: Dataset = struct.field(pytree_node=False)

        asset: Int[Array, "..."]
        market: Int[Array, "..."]
        start_at: Int[Array, "..."]
        stop_at: Int[Array, "..."]

        @cached_property
        def duration(self):
            return self.stop_at - self.start_at

        @property
        def start_log_price(self):
            return self.dataset.log_price[self.start_at, self.asset, self.market]

        @property
        def stop_log_price(self):
            return self.dataset.log_price[self.stop_at, self.asset, self.market]

        @property
        def returns(self):
            return self.stop_log_price - self.start_log_price

        @property
        def direction(self):
            return jnp.sign(self.returns)

        def __len__(self):
            return len(self.start_at)

        def __getitem__(self, *args):
            return tree.map(lambda v: v.__getitem__(*args), self)

    @struct.dataclass
    class Labelset(struct.PyTreeNode):
        dataset: Dataset
        tags: Float[Array, "time asset market"]
        trends: dict[tuple[int, int], Trends]
        events: dict[tuple[int, int], Events]

        @classmethod
        def from_dataset_tags(cls, dataset: Dataset, tags: Float[Array, "time ..."]):
            events = jnp.diff(jnp.sign(tags), axis=0, append=tags[-1:]) / 2

            t_idx, a_idx, m_idx = jnp.nonzero(events)

            def event_fn(a, m):
                at = t_idx[(a_idx == a) & (m_idx == m)]
                return Events(
                    dataset=dataset,
                    asset=jnp.full_like(at, a),
                    market=jnp.full_like(at, m),
                    at=at,
                )

            events = {
                (a, m): event_fn(a, m)
                for a in range(tags.shape[1])
                for m in range(tags.shape[2])
            }

            def trend_fn(a, m):
                start_idx = t_idx[(a_idx == a) & (m_idx == m)]
                start_idx, stop_idx = start_idx[:-1], start_idx[1:]
                return Trends(
                    dataset=dataset,
                    asset=jnp.full_like(stop_idx, a),
                    market=jnp.full_like(stop_idx, m),
                    start_at=start_idx,
                    stop_at=stop_idx,
                )

            trends = {
                (a, m): trend_fn(a, m)
                for a in range(tags.shape[1])
                for m in range(tags.shape[2])
            }

            return cls(dataset=dataset, tags=tags, events=events, trends=trends)


    ls = Labelset.from_dataset_tags(dataset, labels)
    tr = ls.trends[(0,0)]

    # %%
    import pickle
    data = pickle.dumps(ls)
    lss = pickle.loads(data)

