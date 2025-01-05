"""
This module implements investment portfolio simulation using unsupervised learning.

The simulation functions take a tensor of log asset prices and parameters that
define the portfolio behavior. The simulation model is trained using an optimization
algorithm that minimizes the loss function, defined as the difference between
the expected return of the simulated portfolio and a reference portfolio.
"""

from typing import Optional, Tuple
from jax import numpy as jnp
from flax import nnx, struct
from dataclasses import dataclass
from jax import lax

from neuralab.nl.common import Loss, Metric


@dataclass
class RiskControl:
    initial_margin: float = 0.1  # 10% initial margin
    maint_margin: float = 0.05  # 5% maintenance margin
    max_leverage: float = 5.0  # 5x maximum leverage
    var_limit: float = 0.02  # 2% VaR limit
    max_drawdown_limit: float = 0.15  # 15% max drawdown limit
    confidence_level: float = 0.95


@struct.dataclass
class SpotMetrics(struct.PyTreeNode):
    weights: jnp.ndarray
    log_costs: jnp.ndarray
    portfolio_returns: jnp.ndarray

@struct.dataclass
class FuturesMetrics(struct.PyTreeNode):
    weights: jnp.ndarray
    log_costs: jnp.ndarray
    portfolio_returns: jnp.ndarray
    var: jnp.ndarray
    max_drawdown: jnp.ndarray
    risk_scaling: jnp.ndarray

def calculate_risk_metrics(
    returns: jnp.ndarray, confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Calculate VaR and max drawdown from log returns"""
    var = -jnp.percentile(returns, (1 - confidence_level) * 100)
    cumulative = jnp.cumsum(returns)
    running_max = lax.cummax(cumulative, axis=0)
    drawdown = running_max - cumulative 
    max_drawdown = jnp.max(drawdown)
    return var, max_drawdown


def _calculate_transaction_costs(
    weights: jnp.ndarray,
    transaction_cost: float,
) -> jnp.ndarray:
    """Calculate log transaction costs from weight changes"""
    weight_changes = jnp.abs(weights[1:] - weights[:-1])
    total_turnover = jnp.sum(weight_changes, axis=-1)
    return jnp.log1p(-total_turnover * transaction_cost)


def simulate_spot(
    returns: jnp.ndarray,
    weights: jnp.ndarray,
    transaction_cost: float,
) -> SpotMetrics:
    """
    Simulate basic portfolio without risk control

    Args:
        returns: Array of shape (time_steps, num_assets) containing returns
        weights: Array of shape (time_steps, num_assets) or (num_assets,)
        transaction_cost: Cost as fraction of traded value
    """
    log_costs = _calculate_transaction_costs(weights, transaction_cost)
    portfolio_returns = jnp.sum(returns * weights[:-1], axis=-1) + log_costs

    return SpotMetrics(
        weights=weights,
        log_costs=log_costs,
        portfolio_returns=portfolio_returns
    )


def simulate_futures(
    returns: jnp.ndarray,
    weights: jnp.ndarray,
    transaction_cost: float,
    risk_control: RiskControl,
) -> FuturesMetrics:
    """
    Simulate futures portfolio with risk control and leverage
    """
    if weights.ndim == 1:
        weights = jnp.tile(weights, (len(returns), 1))

    # Apply leverage limits
    active_weights = jnp.clip(
        weights, -risk_control.max_leverage, risk_control.max_leverage
    )

    # Calculate base returns with costs
    log_costs = _calculate_transaction_costs(active_weights, transaction_cost)
    portfolio_returns = jnp.sum(returns * active_weights[:-1], axis=-1) + log_costs

    # Apply risk control
    var, max_drawdown = calculate_risk_metrics(
        portfolio_returns, risk_control.confidence_level
    )

    risk_scaling = jnp.where(
        (var > risk_control.var_limit) | (max_drawdown > risk_control.max_drawdown_limit),
        risk_control.maint_margin / risk_control.initial_margin,
        1.0
    )

    return FuturesMetrics(
        weights=active_weights,
        log_costs=log_costs,
        portfolio_returns=portfolio_returns * risk_scaling,
        var=var,
        max_drawdown=max_drawdown,
        risk_scaling=risk_scaling
    )



def spot_loss(
    metrics: SpotMetrics,
    target_returns: Optional[jnp.ndarray] = None,
) -> Loss:
    """Calculate loss between simulated and target portfolio in log-space"""

    if target_returns is None:
        return Loss(-jnp.mean(metrics.portfolio_returns) / (jnp.std(metrics.portfolio_returns) + 1e-6))

    return Loss(jnp.mean((metrics.portfolio_returns - target_returns) ** 2))


def futures_loss(
    spot_metrics: SpotMetrics,
    futures_metrics: FuturesMetrics,
    target_spot_returns: Optional[jnp.ndarray] = None,
    target_future_returns: Optional[jnp.ndarray] = None,
) -> Loss:
    """Combined portfolio loss with risk control"""
    combined_returns = spot_metrics.portfolio_returns + futures_metrics.portfolio_returns

    target_returns = (
        target_spot_returns + target_future_returns
        if target_spot_returns is not None
        else None
    )

    base_loss = spot_loss(SpotMetrics(
        weights=spot_metrics.weights,
        log_costs=spot_metrics.log_costs,
        portfolio_returns=combined_returns
    ), target_returns)

    risk_penalty = futures_metrics.var + futures_metrics.max_drawdown

    return Loss(base_loss.value + risk_penalty)


class Sim(nnx.Module):
    """Portfolio simulation model operating in log-space"""

    def __init__(
        self,
        transaction_cost: float,
        risk_control: RiskControl = RiskControl(),
    ):
        """
        Args:
            transaction_cost: Cost as fraction of traded value
            risk_control: Optional risk control parameters
        """
        self.transaction_cost = transaction_cost
        self.risk_control = risk_control
        self.training = False

    def __call__(
        self,
        spot_returns: jnp.ndarray,
        futures_returns: jnp.ndarray,
        spot_weights: jnp.ndarray,
        futures_weights: jnp.ndarray,
    ) -> Tuple[SpotMetrics, FuturesMetrics]:
        """
        Simulate combined spot and futures portfolio

        Args:
            spot_returns: Returns for spot market assets
            futures_returns: Returns for futures contracts
            spot_weights: Portfolio weights for spot positions
            futures_weights: Portfolio weights for futures positions

        Returns:
            Tuple of (spot_portfolio_returns, futures_portfolio_returns)
        """
        spot_metrics = simulate_spot(
            spot_returns, spot_weights, self.transaction_cost
        )

        futures_metrics = simulate_futures(
            futures_returns, futures_weights, self.transaction_cost, self.risk_control
        )

        if self.training:
            self.spot_loss = spot_loss(spot_metrics)
            self.futures_loss = futures_loss(spot_metrics, futures_metrics)

        return spot_metrics, futures_metrics
