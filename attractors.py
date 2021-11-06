from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import cos, sin
from typing import Callable, Dict, Tuple

import numpy as np
from numba import jit

""" These define the update equations """


@jit(nopython=True)
def trajectory_steps_2d(
    step: Callable,
    x_0: np.array,
    N_steps: int,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
) -> np.array:
    x = np.zeros((N_steps, x_0.shape[0]))
    x[0] = x_0
    for i in np.arange(N_steps - 1):
        _x = step(x[i, 0], x[i, 1], a, b, c, d, e, f)
        x[i + 1, 0] = _x[0]
        x[i + 1, 1] = _x[1]
    return x


@jit(nopython=True)
def trajectory_steps_3d(
    step: Callable,
    x_0: np.array,
    N_steps: int,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
) -> np.ndarray:
    pass


@jit(nopython=True)
def clifford_step(
    x: float, y: float, a: float, b: float, c: float, d: float, *args
) -> Tuple[float]:
    return (
        sin(a * y) + c * cos(a * x),
        sin(b * x) + d * cos(b * y),
    )


@jit(nopython=True)
def de_jong_step(
    x: float, y: float, a: float, b: float, c: float, d: float, *args
) -> Tuple[float]:
    return (
        sin(a * y) - cos(b * x),
        sin(c * x) - cos(d * y),
    )


@jit(nopython=True)
def ikeda_step(
    x: float, y: float, a: float, b: float, c: float, d: float, *args
) -> Tuple[float]:
    _tmp = (a - c) / (1 + (x ** 2) + (y ** 2))
    return (
        d + b * ((x * np.cos(_tmp)) - (y * np.sin(_tmp))),
        b * ((x * np.sin(_tmp)) + (y * np.cos(_tmp))),
    )


@dataclass
class Attractor:
    name: str
    step: Callable
    N: int
    default_params: Dict[str, float]


class Attractors(Enum):
    CLIFFORD = Attractor(
        "clifford", clifford_step, 2, {"a": 1.5, "b": -1.8, "c": 1.6, "d": 2}
    )
    DE_JONG = Attractor(
        "de_jong", de_jong_step, 2, {"a": -0.709, "b": 1.638, "c": 0.452, "d": 1.740}
    )
    IKEDA = Attractor("ikeda", ikeda_step, 2, {"a": 0.4, "b": 0.9, "c": 6, "d": 1})
    LORENZ = Attractor("lorenz", lambda x: x, 3, {"a": 10, "b": 28, "c": 8 / 3})
    ROSSLER = Attractor("rossler", lambda x: x, 3, {"a": 0.1, "b": 0.1, "c": 14})


# # Euler's method diverge?
# def lorentz_attractor_derivative(X, params):
#     _X = np.zeros_like(X)
#     _X[0] = params["a"] * (X[1] - X[0])
#     _X[1] = X[0] * (params["b"] - X[2]) - X[1]
#     _X[2] = (X[0] * X[1]) - (params["c"] * X[2])
#     return _X


# def lorentz_attractor(X, params=LORENTZ_DEFAULT_PARAMS, step_size=1e-3):
#     return X + (lorentz_attractor_derivative(X, params=params) * step_size)


# def rossler_attractor_derivative(X, params):
#     _X = np.zeros_like(X)
#     _X[0] = -X[1] - X[2]
#     _X[1] = X[0] + params["a"] * X[1]
#     _X[2] = params["b"] + X[2] * (X[0] - params["c"])
#     return _X


# def rossler_attractor(X, params=ROSSLER_DEFAULT_PARAMS, step_size=1e-3):
#     return X + (rossler_attractor_derivative(X, params=params) * step_size)
