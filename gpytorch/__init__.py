#!/usr/bin/env python3
from . import (
    beta_features,
    distributions,
    kernels,
    lazy,
    likelihoods,
    means,
    mlls,
    models,
    priors,
    settings,
    utils,
    variational,
)
from .functions import (  # Deprecated
    add_diag,
    add_jitter,
    dsmm,
    inv_matmul,
    inv_quad,
    inv_quad_log_det,
    inv_quad_logdet,
    log_det,
    log_normal_cdf,
    logdet,
    matmul,
    root_decomposition,
    root_inv_decomposition,
)
from .lazy import cat, delazify, lazify
from .mlls import ExactMarginalLogLikelihood
from .module import Module

__version__ = "0.3.6"

__all__ = [
    # Submodules
    "distributions",
    "kernels",
    "lazy",
    "likelihoods",
    "means",
    "mlls",
    "models",
    "priors",
    "utils",
    "variational",
    # Classes
    "Module",
    "ExactMarginalLogLikelihood",
    # Functions
    "add_diag",
    "add_jitter",
    "cat",
    "delazify",
    "dsmm",
    "inv_matmul",
    "inv_quad",
    "inv_quad_logdet",
    "lazify",
    "logdet",
    "log_normal_cdf",
    "matmul",
    "root_decomposition",
    "root_inv_decomposition",
    # Context managers
    "beta_features",
    "settings",
    # Other
    "__version__",
    # Deprecated
    "inv_quad_log_det",
    "log_det",
]
