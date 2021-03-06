import numpy as np

""" These define the update equations """

# 2D
CLIFFORD = "clifford"
CLIFFORD_DEFAULT_PARAMS = {"a": 1.5, "b": -1.8, "c": 1.6, "d": 2}
def clifford_attractor(X, params=CLIFFORD_DEFAULT_PARAMS):
    _X = np.zeros_like(X)
    _X[0] = np.sin(params["a"] * X[1]) + params["c"] * np.cos(params["a"] * X[0])
    _X[1] = np.sin(params["b"] * X[0]) + params["d"] * np.cos(params["b"] * X[1])
    return _X

DE_JONG = "de_jong"
DE_JONG_DEFAULT_PARAMS = {"a": -0.709, "b": 1.638, "c":0.452, "d":1.740}
def de_jong_attractor(X, params=DE_JONG_DEFAULT_PARAMS):
    _X = np.zeros_like(X)
    _X[0] = np.sin(params["a"] * X[1]) - np.cos(params["b"] * X[0])
    _X[1] = np.sin(params["c"] * X[0]) - np.cos(params["d"] * X[1])
    return _X

IKEDA = "ikeda"
IKEDA_DEFAULT_PARAMS = params={"a": 0.4, "b": 0.9, "c": 6, "d": 1}
def ikeda_attractor(X, params=IKEDA_DEFAULT_PARAMS):
    _X = np.zeros_like(X)
    temp = (params["a"] - params["c"]) / (1 + (X[0] * X[0]) + (X[1] * X[1])) # Parenthesis or not ?
    _X[0] = params["d"] + params["b"] * ((X[0] * np.cos(temp)) - (X[1] * np.sin(temp)))
    _X[1] = params["b"] * ((X[0] * np.sin(temp)) + (X[1] * np.cos(temp)))
    return _X

# 3D
LORENTZ = "lorentz"
LORENTZ_DEFAULT_PARAMS = {"a": 10, "b": 28, "c": 8/3}
# Euler's method diverge?
def lorentz_attractor_derivative(X, params):
    _X = np.zeros_like(X)
    _X[0] = params["a"] * (X[1] - X[0])
    _X[1] = X[0] * (params["b"] - X[2]) - X[1]
    _X[2] = (X[0] * X[1]) - (params["c"] * X[2])
    return _X

def lorentz_attractor(X, params=LORENTZ_DEFAULT_PARAMS, step_size=1e-3):
    return X + (lorentz_attractor_derivative(X, params=params) * step_size)

ROSSLER = "rossler"
ROSSLER_DEFAULT_PARAMS = {"a": 0.1, "b": 0.1, "c": 14}
def rossler_attractor_derivative(X, params):
    _X = np.zeros_like(X)
    _X[0] = - X[1] - X[2]
    _X[1] = X[0] + params["a"] * X[1]
    _X[2] = params["b"] + X[2] * (X[0] - params["c"])
    return _X

def rossler_attractor(X, params=ROSSLER_DEFAULT_PARAMS, step_size=1e-3):
    return X + (rossler_attractor_derivative(X, params=params) * step_size)


# %
def get_attractor_function(attractor, params):
    if attractor == CLIFFORD:
        attractor_f = clifford_attractor
        params = params if params is not None else CLIFFORD_DEFAULT_PARAMS
    elif attractor == DE_JONG:
        attractor_f = de_jong_attractor
        params = params if params is not None else DE_JONG_DEFAULT_PARAMS
    elif attractor == IKEDA:
        attractor_f = ikeda_attractor
        params = params if params is not None else IKEDA_DEFAULT_PARAMS
    elif attractor == LORENTZ:
        attractor_f = lorentz_attractor
        params = params if params is not None else LORENTZ_DEFAULT_PARAMS
    elif attractor == ROSSLER:
        attractor_f = rossler_attractor
        params = params if params is not None else ROSSLER_DEFAULT_PARAMS
    
    return attractor_f, params