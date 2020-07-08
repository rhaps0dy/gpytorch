import math
import warnings

import numpy.polynomial.hermite
import torch

from ..distributions import base_distributions
from ..constraints import LessThan, ConstraintViolationError
from ..lazy import BlockInterleavedLazyTensor
from .softmax_likelihood import SoftmaxLikelihood


class RobustmaxLikelihood(SoftmaxLikelihood):
    "Robustmax likelihood for multiclass GP classification."
    def __init__(self, num_classes, epsilon=1e-3,
                 epsilon_constraint=None,
                 num_quadrature_points=20):
        super().__init__(num_classes=num_classes, mixing_weights=False)
        self.num_quadrature_points = num_quadrature_points
        if log_epsilon_constraint is None:
            log_epsilon_constraint = LessThan(- 2**-20)

        self.register_parameter("raw_log_epsilon", torch.nn.Parameter(torch.zeros(())))
        self.register_constraint("raw_log_epsilon", log_epsilon_constraint)
        self.epsilon = epsilon  # Set raw parameter

    @property
    def epsilon(self):
        return self.log_epsilon.exp()

    @epsilon.setter
    def epsilon(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_log_epsilon)
        self.log_epsilon = value.log()

    @property
    def log_epsilon(self):
        return self.raw_log_epsilon_constraint.transform(self.raw_log_epsilon)

    @log_epsilon.setter
    def log_epsilon(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_log_epsilon)
        try:
            self.initialize(raw_log_epsilon=self.raw_log_epsilon_constraint.inverse_transform(value))
        except ConstraintViolationError:
            raise ConstraintViolationError(
                "Attempting to manually set a parameter value that is out of bounds of "
                f"its current constraints, {self.raw_log_epsilon}. "
                "Most likely, you want to do the following:\n likelihood = RobustmaxLikelihood"
                "(log_epsilon_constraint=gpytorch.constraints.LessThan(better_upper_bound))"
            )

    def log_class_probs(self):
        "returns log(epsilon/(num_classes-1)), log(1-epsilon)"
        low_log_prob = self.log_epsilon - math.log(self.num_classes-1)
        high_log_prob = (1 - self.epsilon).log()
        return low_log_prob, high_log_prob

    def forward(self, function_samples, *params, **kwargs):
        if function_samples.requires_grad:
            warnings.warn(
                "RobustmaxLikelihood.forward is not differentiable, but the "
                "input to it has requires_grad=True",
                RuntimeWarning)

        num_data, num_features = function_samples.shape[-2:]
        # Catch legacy mode
        if num_data == self.num_features:
            warnings.warn(
                "The input to SoftmaxLikelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated."
            )
            function_samples = function_samples.transpose(-1, -2)
            num_data, num_features = function_samples.shape[-2:]

        if num_features != self.num_classes:
            raise RuntimeError(
                f"There should be {self.num_classes} features, as many as "
                "classes")

        one_hot = torch.nn.functional.one_hot(function_samples.argmax(-1),
                                              num_classes=self.num_classes)
        # one_hot is either 0 or 1, indexes into this 2-element tensor
        low_high_log_probs = torch.stack(self.log_class_probs())
        res = base_distributions.Categorical(logits=low_high_log_probs[one_hot])
        return res

    def expected_log_prob(self, observations, function_dist, *params, **kwargs):
        prob_is_largest = self._prob_is_largest(observations, function_dist, *params, **kwargs)
        low_log_prob, high_log_prob = self.log_class_probs()
        return high_log_prob*prob_is_largest + low_log_prob*(1-prob_is_largest)

    def _prob_is_largest(self, observations, function_dist, *params, **kwargs):
        if len(params) > 0 or len(kwargs) > 0:
            raise ValueError(f"Not expecting any params={params}, or kwargs={kwargs}")
        lazy_cov = function_dist.lazy_covariance_matrix
        if isinstance(lazy_cov, BlockInterleavedLazyTensor):
            means = function_dist.mean
            stds = function_dist.stddev
        else:
            if function_dist.mean.size(-2) == self.num_classes:
                # Legacy mode
                means = function_dist.mean.transpose(-1, -2)
                stds = function_dist.stddev.transpose(-1, -2)
            else:
                raise NotImplementedError(
                    "The task dimension of covariance matrix"
                    f"{function_dist.lazy_covariance_matrix} may not be independent")
        assert stds.size()[-2:] == means.size()[-2:]
        # S = possible batch size of `means`, `stds`.
        # N = number of observations
        # K = self.num_classes
        # H = self.num_quadrature_points

        label_idx = observations.unsqueeze(-1)                        #   N,1
        label_idx = label_idx.repeat(
            *means.shape[:-label_idx.dim()],
            *([1] * label_idx.dim()))                                 # S,N,1
        true_class_means = torch.gather(
            input=means, dim=-1, index=label_idx)                     # S,N,1
        true_class_stds = torch.gather(
            input=stds, dim=-1, index=label_idx)                      # S,N,1

        stds_repr = stds.reciprocal()
        normalised_means = (true_class_means - means).mul(stds_repr)  # S,N,K
        # sqrt(2) is for the quadrature x
        normalised_stds = math.sqrt(2) * (true_class_stds*stds_repr)  # S,N,K

        quad_x, quad_w = map(
            lambda t: torch.from_numpy(t).to(means),
            numpy.polynomial.hermite.hermgauss(self.num_quadrature_points))
        # quad_w has to be multiplied by π^(-D/2)
        quad_w /= math.sqrt(math.pi)                                  #     H

        x = quad_x.unsqueeze(-1)\
                  .mul(normalised_stds.unsqueeze(-2))\
                  .add(normalised_means.unsqueeze(-2))                # S,N,H,K
        # 1/sqrt(2) constant is because we use erf
        # GaussCDF(x) = .5 * (1 + erf(x / sqrt(2)))
        cdfs = .5 * (1 + torch.erf(x / math.sqrt(2)))                 # S,N,H,K

        mask = torch.nn.functional.one_hot(observations, self.num_classes)\
                                  .to(torch.bool).unsqueeze(-2)       # S,N,1,K
        cdfs = cdfs.masked_fill(mask, 1.)   # so that it does not affect the product

        # unsqueeze: work around https://github.com/pytorch/pytorch/issues/38315
        sample_elp = cdfs.prod(dim=-1) @ quad_w.unsqueeze(-1)         # S,N,1
        return sample_elp.squeeze(-1)
