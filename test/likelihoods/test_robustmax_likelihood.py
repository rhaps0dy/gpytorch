#!/usr/bin/env python3

import unittest

import scipy.stats
import torch
from torch.distributions import Distribution

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import RobustmaxLikelihood
from gpytorch.test.base_likelihood_test_case import BaseLikelihoodTestCase
import math
import numpy as np

def robustmax(train_y, n_classes, means, stds, H=20):
    label_idx = train_y.unsqueeze(-1)                   #   N,1
    label_idx = label_idx.repeat(
        *means.shape[:-label_idx.dim()],
        *([1] * label_idx.dim()))                       # S,N,1
    true_class_means = torch.gather(
        input=means, dim=-1, index=label_idx)           # S,N,1

    # 1/sqrt(2) constant is because we use erf
    # GaussCDF(x) = .5 * (1 + erf(x / sqrt(2)))
    normalised_means = (true_class_means - means).mul(
        stds.reciprocal())               # S,N,K

    quad_x, quad_w = map(
        lambda t: torch.from_numpy(t).to(means.dtype),
        np.polynomial.hermite.hermgauss(H))
    # quad_w has to be multiplied by π^(-D/2)
    quad_w /= math.sqrt(math.pi)                    #       H

    x = quad_x.unsqueeze(-1) + normalised_means.unsqueeze(-2)
    cdfs = .5 * (1 + torch.erf(x / np.sqrt(2)))

    mask = torch.nn.functional.one_hot(train_y, n_classes).to(torch.bool).unsqueeze(-2)
    cdfs = cdfs.masked_fill(mask, 1.)   # so that it does not affect the product

    sample_lik = cdfs.prod(dim=-1) @ quad_w

    return sample_lik


class TestRobustmaxLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 0

    def _create_conditional_input(self, batch_shape=torch.Size([])):
        return torch.randn(*batch_shape, 4, 5)

    def _create_marginal_input(self, batch_shape=torch.Size([])):
        mat = torch.randn(*batch_shape, 4, 5, 5)
        cov = mat @ mat.transpose(-1, -2)
        cov = cov.add_(torch.eye(5)*0.001)
        return MultivariateNormal(torch.randn(*batch_shape, 4, 5), cov)

    def _create_targets(self, batch_shape=torch.Size([])):
        return torch.distributions.Categorical(probs=torch.tensor([0.25, 0.25, 0.25, 0.25])).sample(
            torch.Size([*batch_shape, 5])
        )

    def create_likelihood(self, num_classes=4, **kwargs):
        return RobustmaxLikelihood(num_classes=num_classes, **kwargs)

    def _test_conditional(self, batch_shape):
        likelihood = self.create_likelihood()
        input = self._create_conditional_input(batch_shape)
        output = likelihood(input)

        self.assertIsInstance(output, Distribution)
        self.assertEqual(output.sample().shape, torch.Size([*batch_shape, 5]))

    def _test_log_prob(self, batch_shape):
        likelihood = self.create_likelihood()
        input = self._create_marginal_input(batch_shape)
        target = self._create_targets(batch_shape)
        output = likelihood.expected_log_prob(target, input)

        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, batch_shape + torch.Size([5]))

    def _test_marginal(self, batch_shape):
        likelihood = self.create_likelihood()
        input = self._create_marginal_input(batch_shape)
        output = likelihood(input)

        self.assertTrue(isinstance(output, Distribution))
        self.assertEqual(output.sample().shape[-len(batch_shape) - 1 :], torch.Size([*batch_shape, 5]))

    def test_prob_is_largest(self, num_classes=3, num_points=1, num_samples=1000000):
        likelihood = self.create_likelihood(num_classes, num_quadrature_points=50)
        input = self._create_marginal_input()

        input = input[:num_classes, :num_points]
        input = input[..., None][..., 0]  # Make the data-wise covariance diagonal
        target = input.sample().argmax(-2)
        output = likelihood._prob_is_largest(target, input).prod()
        # output = robustmax(target, num_classes, input.mean.t(), input.stddev.t(), H=40)

        samples = input.sample(torch.Size([num_samples]))
        accepted = (samples.argmax(-2) == target).all(dim=-1).to(torch.float32)

        assert accepted.size() == torch.Size([num_samples])

        _, p_value = scipy.stats.ttest_1samp(accepted, output)
        assert p_value > 0.3, f"Hypothesis 'implementation is correct' was rejected with p={p_value}"



