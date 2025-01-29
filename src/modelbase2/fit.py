"""Parameter Fitting Module for Metabolic Models.

This module provides functions foru fitting model parameters to experimental data,
including both steadyd-state and time-series data fitting capabilities.e

Functions:
    fit_steady_state: Fits parameters to steady-state experimental data
    fit_time_course: Fits parameters to time-series experimental data
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from modelbase2.integrators import DefaultIntegrator
from modelbase2.simulator import Simulator
from modelbase2.types import Array, ArrayLike, Callable, IntegratorProtocol, cast

__all__ = [
    "InitialGuess",
    "MinimizeFn",
    "ResidualFn",
    "SteadyStateResidualFn",
    "TimeSeriesResidualFn",
    "rmse_ss",
    "rmse_tc",
    "steady_state",
    "time_course",
]

if TYPE_CHECKING:
    from modelbase2.model import Model

type InitialGuess = dict[str, float]
type ResidualFn = Callable[[Array], float]
type MinimizeFn = Callable[[ResidualFn, InitialGuess], dict[str, float]]


class SteadyStateResidualFn(Protocol):
    """Protocol for steady state residual functions."""

    def __call__(
        self,
        par_values: Array,
        # This will be filled out by partial
        par_names: list[str],
        data: pd.Series,
        model: Model,
        y0: dict[str, float],
        integrator: type[IntegratorProtocol],
    ) -> float:
        """Calculate residual error between model steady state and experimental data."""
        ...


class TimeSeriesResidualFn(Protocol):
    """Protocol for time series residual functions."""

    def __call__(
        self,
        par_values: Array,
        # This will be filled out by partial
        par_names: list[str],
        data: pd.DataFrame,
        model: Model,
        y0: dict[str, float],
        integrator: type[IntegratorProtocol],
    ) -> float:
        """Calculate residual error between model time course and experimental data."""
        ...


def _default_minimize_fn(
    residual_fn: ResidualFn,
    p0: dict[str, float],
) -> dict[str, float]:
    res = minimize(
        residual_fn,
        x0=list(p0.values()),
        bounds=[(1e-12, 1e6) for _ in range(len(p0))],
        method="L-BFGS-B",
    )
    if res.success:
        return dict(
            zip(
                p0,
                res.x,
                strict=True,
            )
        )
    return dict(zip(p0, np.full(len(p0), np.nan, dtype=float), strict=True))


def rmse_ss(data: pd.Series, results: pd.DataFrame) -> float:
    diff = data - results.loc[:, data.index]
    return cast(float, np.sqrt(np.mean(np.square(diff))))


def rmse_tc(data: pd.DataFrame, results: pd.DataFrame) -> float:
    diff = data - results.loc[:, data.columns]
    return cast(float, np.sqrt(np.mean(np.square(diff))))


def _steady_state_residual(
    par_values: Array,
    # This will be filled out by partial
    par_names: list[str],
    data: pd.Series,
    model: Model,
    y0: dict[str, float] | None,
    integrator: type[IntegratorProtocol],
    likelihood: Callable,
) -> float:
    """Calculate residual error between model steady state and experimental data.

    Args:
        par_values: Parameter values to test
        data: Experimental steady state data
        model: Model instance to simulate
        y0: Initial conditions
        par_names: Names of parameters being fit
        integrator: ODE integrator class to use

    Returns:
        float: Root mean square error between model and data

    """
    c_ss, v_ss = (
        Simulator(
            model.update_parameters(
                dict(
                    zip(
                        par_names,
                        par_values,
                        strict=True,
                    )
                )
            ),
            y0=y0,
            integrator=integrator,
        )
        .simulate_to_steady_state()
        .get_full_concs_and_fluxes()
    )
    if c_ss is None or v_ss is None:
        return cast(float, np.inf)
    results_ss = pd.concat((c_ss, v_ss), axis=1)
    return likelihood(data, results_ss)


def _time_course_residual(
    par_values: ArrayLike,
    # This will be filled out by partial
    par_names: list[str],
    data: pd.DataFrame,
    model: Model,
    y0: dict[str, float],
    integrator: type[IntegratorProtocol],
    likelihood: Callable,
) -> float:
    """Calculate residual error between model time course and experimental data.

    Args:
        par_values: Parameter values to test
        data: Experimental time course data
        model: Model instance to simulate
        y0: Initial conditions
        par_names: Names of parameters being fit
        integrator: ODE integrator class to use

    Returns:
        float: Root mean square error between model and data

    """
    c_ss, v_ss = (
        Simulator(
            model.update_parameters(dict(zip(par_names, par_values, strict=True))),
            y0=y0,
            integrator=integrator,
        )
        .simulate_time_course(data.index)  # type: ignore
        .get_full_concs_and_fluxes()
    )
    if c_ss is None or v_ss is None:
        return cast(float, np.inf)
    results_ss = pd.concat((c_ss, v_ss), axis=1)
    return likelihood(data, results_ss)


def steady_state(
    model: Model,
    p0: dict[str, float],
    data: pd.Series,
    y0: dict[str, float] | None = None,
    minimize_fn: MinimizeFn = _default_minimize_fn,
    residual_fn: SteadyStateResidualFn = _steady_state_residual,
    integrator: type[IntegratorProtocol] = DefaultIntegrator,
    likelihood: Callable = rmse_ss,
) -> dict[str, float]:
    """Fit model parameters to steady-state experimental data.

    Examples:
        >>> steady_state(model, p0, data)
        {'k1': 0.1, 'k2': 0.2}

    Args:
        model: Model instance to fit
        data: Experimental steady state data as pandas Series
        p0: Initial parameter guesses as {parameter_name: value}
        y0: Initial conditions as {species_name: value}
        minimize_fn: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class

    Returns:
        dict[str, float]: Fitted parameters as {parameter_name: fitted_value}

    Note:
        Uses L-BFGS-B optimization with bounds [1e-12, 1e6] for all parameters

    """
    par_names = list(p0.keys())

    # Copy to restore
    p_orig = model.parameters

    fn = cast(
        ResidualFn,
        partial(
            residual_fn,
            data=data,
            model=model,
            y0=y0,
            par_names=par_names,
            integrator=integrator,
            likelihood=likelihood,
        ),
    )
    res = minimize_fn(fn, p0)

    # Restore
    model.update_parameters(p_orig)
    return res


def time_course(
    model: Model,
    p0: dict[str, float],
    data: pd.DataFrame,
    y0: dict[str, float] | None = None,
    minimize_fn: MinimizeFn = _default_minimize_fn,
    residual_fn: TimeSeriesResidualFn = _time_course_residual,
    integrator: type[IntegratorProtocol] = DefaultIntegrator,
    likelihood: Callable = rmse_tc,
) -> dict[str, float]:
    """Fit model parameters to time course of experimental data.

    Examples:
        >>> time_course(model, p0, data)
        {'k1': 0.1, 'k2': 0.2}

    Args:
        model: Model instance to fit
        data: Experimental time course data as pandas DataFrame
        p0: Initial parameter guesses as {parameter_name: value}
        y0: Initial conditions as {species_name: value}
        minimize_fn: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class

    Returns:
        dict[str, float]: Fitted parameters as {parameter_name: fitted_value}

    Note:
        Uses L-BFGS-B optimization with bounds [1e-12, 1e6] for all parameters

    """
    par_names = list(p0.keys())
    p_orig = model.parameters

    fn = cast(
        ResidualFn,
        partial(
            residual_fn,
            data=data,
            model=model,
            y0=y0,
            par_names=par_names,
            integrator=integrator,
            likelihood=likelihood,
        ),
    )
    res = minimize_fn(fn, p0)
    model.update_parameters(p_orig)
    return res
