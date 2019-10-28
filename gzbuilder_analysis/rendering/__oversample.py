import numpy as np
try:
    from cupy import get_array_module
except ModuleNotFoundError:
    def get_array_module(*args, **kwargs):
        return np


def oversampled_function(func):
    _p = get_array_module(func(0, 0))
    def _new_func(*args, shape=(256, 256), oversample_n=5, **kwargs):
        n = oversample_n
        x = _p.linspace(0.5 / n - 0.5, shape[1] - 0.5 - 0.5 / n, shape[1] * n)
        y = _p.linspace(0.5 / n - 0.5, shape[0] - 0.5 - 0.5 / n, shape[0] * n)
        cx, cy = _p.meshgrid(x, y)
        oversampled_comp = func(cx, cy, *args, **kwargs) \
            .reshape(shape[0], oversample_n, shape[1], oversample_n) \
            .mean(3).mean(1)
        return oversampled_comp
    return _new_func
