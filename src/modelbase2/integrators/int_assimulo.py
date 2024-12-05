"""Assimulo integrator for solving ODEs."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Assimulo",
]

import contextlib
import os
from typing import TYPE_CHECKING, Literal

import numpy as np

with contextlib.redirect_stderr(open(os.devnull, "w")):
    from assimulo.problem import Explicit_Problem  # type: ignore
    from assimulo.solvers import CVode  # type: ignore
    from assimulo.solvers.sundials import CVodeError  # type: ignore

if TYPE_CHECKING:
    from collections.abc import Callable

    from modelbase2.types import ArrayLike


@dataclass
class Assimulo:
    """Assimulo integrator for solving ODEs.

    Attributes:
        rhs: Right-hand side function of the ODE.
        y0: Initial conditions.
        atol: Absolute tolerance for the solver.
        rtol: Relative tolerance for the solver.
        maxnef: Maximum number of error failures.
        maxncf: Maximum number of convergence failures.
        verbosity: Verbosity level of the solver.

    Methods:
        integrate: Integrate the ODE system.

    """

    rhs: Callable
    y0: ArrayLike
    atol: float = 1e-8
    rtol: float = 1e-8
    maxnef: int = 4  # max error failures
    maxncf: int = 1  # max convergence failures
    verbosity: Literal[50, 40, 30, 20, 10] = 50

    def __post_init__(self) -> None:
        """Post-initialization method for setting up the CVode integrator with the provided parameters.

        This method initializes the CVode integrator with an explicit problem defined by the
        right-hand side function (`self.rhs`) and the initial conditions (`self.y0`). It also
        sets various integrator options such as absolute tolerance (`self.atol`), relative
        tolerance (`self.rtol`), maximum number of error test failures (`self.maxnef`), maximum
        number of convergence failures (`self.maxncf`), and verbosity level (`self.verbosity`).

        """
        self.integrator = CVode(Explicit_Problem(self.rhs, self.y0))
        self.integrator.atol = self.atol
        self.integrator.rtol = self.rtol
        self.integrator.maxnef = self.maxnef
        self.integrator.maxncf = self.maxncf
        self.integrator.verbosity = self.verbosity

    def reset(self) -> None:
        """Reset the integrator."""
        self.integrator.reset()

    def integrate(
        self,
        *,
        t_end: float | None = None,
        steps: int | None = None,
        time_points: ArrayLike | None = None,
    ) -> tuple[ArrayLike | None, ArrayLike | None]:
        """Integrate the ODE system.

        Args:
            t_end: Terminal time point for the integration.
            steps: Number of steps for the integration.
            time_points: Time points for the integration.

        Returns:
            np.ndarray: Array of integrated values.

        """
        if steps is None:
            steps = 0
        try:
            return self.integrator.simulate(t_end, steps, time_points)  # type: ignore
        except CVodeError:
            return None, None

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
        t_max: float = 1_000_000_000,
    ) -> tuple[float | None, ArrayLike | None]:
        """Integrate the ODE system to steady state.

        Args:
            tolerance: Tolerance for determining steady state.
            rel_norm: Whether to use relative normalization.
            t_max: Maximum time point for the integration (default: 1,000,000,000).

        Returns:
            tuple[float | None, ArrayLike | None]: Tuple containing the final time point and the integrated values at steady state.

        """
        self.reset()

        try:
            for t_end in np.geomspace(1000, t_max, 3):
                t, y = self.integrator.simulate(t_end)
                diff = (y[-1] - y[-2]) / y[-1] if rel_norm else y[-1] - y[-2]
                if np.linalg.norm(diff, ord=2) < tolerance:
                    return t[-1], y[-1]
        except CVodeError:
            return None, None
        return None, None
