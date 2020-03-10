try:
    from .jax.sersic import sersic, oversampled_sersic_component
except ModuleNotFoundError:
    try:
        from .numba.sersic import sersic, oversampled_sersic_component
    except ModuleNotFoundError:
        from .numpy.sersic import sersic, oversampled_sersic_component
