"""The identity the whole Rc-PGM construction rests on.

The reduction map R sends a unit vector to its image in the symmetric
subspace, of dimension d_sym = C(d + c - 1, c) instead of d^c. The claim that
makes the reduction exact rather than approximate is

    <R(x), R(z)> = <x, z>^c

for every pair of vectors. If that fails, the Rc-PGM is not the c-PGM and no
timing comparison in the paper means anything. It is checked here against the
closed form and, for small dimensions where the tensor power is affordable,
against an explicitly constructed x^{tensor c}.

The numpy stand-in of the harness carries the same reduction as the published
estimator, so this runs without PyTorch.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest


def reducer(harness, d: int, c: int):
    """A mock estimator with its symmetric basis built for dimension d."""
    est = harness._MockPGM(n_copies=c)
    est.d = d
    est.occupation_numbers = sorted(est._enumerate_occupation_numbers(), reverse=True)
    est.dsym = len(est.occupation_numbers)
    est.multinomial_factors = est._calculate_multinomial_factors()
    return est


def unit_rows(rng, n: int, d: int) -> np.ndarray:
    X = rng.normal(size=(n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.mark.parametrize("d", [2, 3, 4, 5])
@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_reduction_preserves_the_c_th_power_of_the_inner_product(harness, d, c):
    rng = np.random.default_rng(0)
    X = unit_rows(rng, 12, d)

    reduced = reducer(harness, d, c).map_batch_efficiently(X)
    gram_reduced = reduced @ reduced.T
    gram_expected = (X @ X.T) ** c

    np.testing.assert_allclose(gram_reduced, gram_expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("d", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("c", [1, 2, 3, 5])
def test_reduced_dimension_is_the_binomial(harness, d, c):
    assert reducer(harness, d, c).dsym == math.comb(d + c - 1, c)


@pytest.mark.parametrize("d,c", [(2, 3), (3, 2), (3, 3), (4, 2)])
def test_reduction_agrees_with_the_explicit_tensor_power(harness, d, c):
    """The closed form is checked against x tensored with itself c times."""
    rng = np.random.default_rng(1)
    X = unit_rows(rng, 6, d)

    def tensor_power(x):
        out = np.ones(d**c)
        for k, index in enumerate(product(range(d), repeat=c)):
            out[k] = np.prod([x[i] for i in index])
        return out

    explicit = np.stack([tensor_power(x) for x in X])
    reduced = reducer(harness, d, c).map_batch_efficiently(X)

    np.testing.assert_allclose(
        reduced @ reduced.T, explicit @ explicit.T, rtol=0, atol=1e-12
    )


def test_identity_survives_negative_components_and_exact_zeros(harness):
    """The signed integer powers are where a naive implementation goes wrong."""
    d, c = 4, 3
    X = np.array(
        [
            [0.6, -0.8, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [-0.5, 0.5, -0.5, 0.5],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    reduced = reducer(harness, d, c).map_batch_efficiently(X)
    np.testing.assert_allclose(reduced @ reduced.T, (X @ X.T) ** c, rtol=0, atol=1e-12)


def test_occupation_numbers_are_a_partition_of_the_copies(harness):
    """Every basis vector distributes exactly c copies over the d modes."""
    d, c = 5, 4
    est = reducer(harness, d, c)
    assert len({tuple(t) for t in est.occupation_numbers}) == est.dsym
    assert all(sum(t) == c and len(t) == d for t in est.occupation_numbers)
