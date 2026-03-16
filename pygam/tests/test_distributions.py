"""Tests for pygam.distributions."""

import numpy as np

from pygam.distributions import InvGaussDist


class TestInvGaussDistLogPdf:
    """Tests for InvGaussDist.log_pdf parameterization."""

    @staticmethod
    def _closed_form_logpdf(y, mu, lam):
        return 0.5 * (np.log(lam) - np.log(2 * np.pi) - 3 * np.log(y)) - (
            lam * (y - mu) ** 2
        ) / (2 * mu**2 * y)

    def test_control_case_matches_closed_form(self):
        dist = InvGaussDist(scale=1.0)
        y = np.array([2.0])
        mu = np.array([2.0])
        w = np.array([1.0])

        result = dist.log_pdf(y, mu, weights=w)
        expected = self._closed_form_logpdf(y, mu, lam=w / 1.0)

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_weighted_non_unit_scale_matches_closed_form(self):
        dist = InvGaussDist(scale=2.5)
        y = np.array([0.5, 1.0, 2.0, 4.0])
        mu = np.array([0.6, 1.4, 1.8, 3.2])
        w = np.array([0.5, 1.0, 2.0, 5.0])

        result = dist.log_pdf(y, mu, weights=w)
        expected = self._closed_form_logpdf(y, mu, lam=w / 2.5)

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
