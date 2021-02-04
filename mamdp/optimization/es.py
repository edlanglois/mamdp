"""Evolution Strategies Search Algorithms."""
from __future__ import annotations

import itertools
import math
import numbers
from typing import Callable, Generator, Iterable, Optional, Tuple

import numpy as np
import scipy.linalg
import scipy.stats


def minimize_uh_cma_es(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    step_size: float,
    num_iterations: int,
    population_size: Optional[int] = None,
    parent_population_size: Optional[int] = None,
    num_sample_evaluations: int = 1,
    scale_sample_evaluations: float = 1.5,
    rand: np.random.RandomState = np.random,
):
    """Minimize a function using CMA-ES

    Args:
        f: The function to minimize. Takes a 1D array as input.
        x0: The starting point in the minimization.
        step_size: Initial step size.
        num_iterations: Number of minimization iterations to run.
        population_size: Number of query points per update step.
        parent_population_size: How many of the top samples are used in parameter
            updates.
        num_sample_evaluations: Initial number of evaluations per sample.
        scale_sample_evaluations: Scale factor when adapting num_sample_evaluations.
        rand: Random state used for sampling.

    See `interactive_uh_cma_es` for argument defaults.

    Returns:
        The estimated minimum of `f`. A vector with the same shape and type as `x0`.
    """
    cma_es = interactive_uh_cma_es(
        x0,
        step_size,
        population_size=population_size,
        parent_population_size=parent_population_size,
        num_sample_evaluations=num_sample_evaluations,
        scale_sample_evaluations=scale_sample_evaluations,
    )
    query, i, mean = next(cma_es)
    while i < num_iterations:
        query, i, mean = cma_es.send(f(query))
    return mean


def interactive_uh_cma_es(
    x0: Iterable[float],
    step_size: float,
    population_size: Optional[int] = None,
    parent_population_size: Optional[int] = None,
    num_sample_evaluations: int = 1,
    scale_sample_evaluations: float = 1.5,
    noise_tolerance: float = 0.2,
    rand: np.random.RandomState = np.random,
) -> Generator[Tuple[np.ndarray, int, np.ndarray], float, None]:
    """Interactive implementation of Uncertainty Handling-CMA-ES

    Minimizes the expected value of a stochastic function of an unconstrained
    N-dimensional real vector.

    Args:
        x0: Initial mean vector.
        step_size: Initial step size.
        population_size: Number of query points per update step.
            Defaults to max(4 + floor(3 ln(num_parameters)), 5)
        parent_population_size: How many of the top samples are used in parameter
            updates. Defaults to population_size // 2.
        num_sample_evaluations: Initial number of evaluations per sample.
        scale_sample_evaluations: Scale factor when adapting num_sample_evaluations.
            Set to `None` for no adaptation.
        noise_tolerance: Tolerated noise level when adapting num_sample_evaluations.
        rand: Random state used for sampling.

    Yields:
        query_point: A vector of parameters to test. The caller should send back
            a sample of the function value at these parameters.
        iteration: The current iteration index.
        current_mean: The current parameter mean vector. Represents the best guess for
            the function minimizer so far. Do not modify in-place.

    Reference:
    Verena Heidrich-Meisner and Christian Igel.
    "Uncertainty handling CMA-ES for reinforcement learning."
    Proceedings of the 11th Annual conference on Genetic and evolutionary computation.
    2009.

    This is an implementation of Algorithm 1: rank-μ CMA-ES
    """
    # Inputs in in terms of variables in the paper:
    # x0: θ_init
    # step_size: σ^(0)
    # population_size: λ
    # parent_population_size: μ
    # num_sample_evaluations: n_eval^(0)
    # scale_sample_evaluations: α
    # noise_tolerance: θ

    # Line 1: Initializations
    mean = np.asarray(x0, dtype=float)  # m
    (num_parameters,) = mean.shape  # n
    cov = np.identity(num_parameters)  # C

    if population_size is None:
        population_size = max(4 + int(3 * np.log(num_parameters)), 5)
    if parent_population_size is None:
        parent_population_size = population_size // 2
    num_reevaluations = max(population_size // 10, 2)  # λ_reev

    # w
    weights = np.log(parent_population_size + 1) - np.log(
        np.arange(1, parent_population_size + 1)
    )
    weights /= np.linalg.norm(weights, ord=1)

    mu_eff = 1 / np.sum(np.square(weights))  # Variance effective selection mass
    mu_cov = mu_eff
    # Paper uses μ_{e↑} without definition.
    # By comparing with other CMA-ES references, I think this is mu_eff
    mu_earrow = mu_eff

    # Low-pass filtered evolution paths
    p_c = np.zeros_like(mean)  # Anisotropic evolution path
    p_s = np.zeros_like(mean)  # Isotropic evolution path

    # Time constants for the evolution paths
    c_c = 4 / (num_parameters + 4)
    c_s = (mu_earrow + 2) / (num_parameters + mu_earrow + 3)
    c_cov = 2 / (num_parameters + np.sqrt(2)) ** 2  # Covariance matrix learning rate

    # Damping factor
    d_s = 1 + 2 * np.sqrt(max(0, (mu_earrow - 1) / (num_parameters + 1))) + c_s
    cd_s = c_s / d_s

    # Expected norm of a standard MVN distribution (used in Line 7)
    expected_norm_mvn = scipy.stats.chi.mean(num_parameters)

    adaptive_num_evaluations = not (
        scale_sample_evaluations is None or scale_sample_evaluations <= 1
    )
    for step in itertools.count():  # Line 2
        # Line 3: Sample Population
        population = mean + step_size * rand.multivariate_normal(
            np.zeros(num_parameters), cov, size=population_size
        )

        # Line 4: Evaluate Samples
        values = yield from _cma_es_collect_samples(
            population, num_sample_evaluations, mean, step
        )

        # Line 5a: Uncertainty handling
        if adaptive_num_evaluations:
            num_sample_evaluations = yield from _uncertainty_handling(
                population=population,
                values=values,
                num_sample_evaluations=num_sample_evaluations,
                num_reevaluations=num_reevaluations,
                scale_sample_evaluations=scale_sample_evaluations,
                noise_tolerance=noise_tolerance,
                mean=mean,
                step=step,
            )

        # Line 5b: Select best samples as the parent population; calculate new mean
        parent_population = population[np.argsort(values)[:parent_population_size]]
        old_mean = mean
        mean = weights @ parent_population

        # Line 6: Update p_σ
        sqrt_cov = _sqrtm_psd(cov)
        normalized_step = (mean - old_mean) / step_size
        adjusted_step, _, _, _ = scipy.linalg.lstsq(sqrt_cov, normalized_step)
        p_s = (1 - c_s) * p_s + np.sqrt(c_s * (2 - c_s) * mu_eff) * adjusted_step

        # Line 7: Update step size
        step_size = step_size * np.exp(
            cd_s * (np.linalg.norm(p_s) / expected_norm_mvn - 1)
        )

        # Line 8: Update p_c
        p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * normalized_step
        cov = (
            (1 - c_cov) * cov
            + c_cov / mu_cov * np.outer(p_c, p_c)
            + (
                c_cov
                * (1 - 1 / mu_cov)
                * np.einsum(
                    "i,ij,ik->jk", weights, parent_population, parent_population
                )
            )
        )


def _sqrtm_psd(a: np.ndarray) -> np.ndarray:
    """Square root of a positive semidefinite symmetric matrix."""
    eigvals, eigvects = np.linalg.eigh(a)
    # Eigenvalues should always be nonnegative for a psd matrix
    # but `eigvals` can be negative due to numerical errors.
    np.maximum(eigvals, 0, out=eigvals)
    return eigvects @ np.diag(np.sqrt(eigvals)) @ eigvects.T


def _cma_es_collect_samples(
    population: np.ndarray, num_sample_evaluations: int, mean: np.ndarray, step: int
) -> Generator[Tuple[np.ndarray, int, np.ndarray], float, np.ndarray]:
    """Generator that yields samples and collectiosn evaluations.

    Args:
        population: A set of population samples to evaluate.
        num_sample_evaluations: Number of evaluations to collect for each sample.
        mean: The current mean vector from which population was sampled.
        step: The current step index.

    Yields:
        sample: The sample to evaluate.
        step: The current step index.
        mean: The current mean vector from which population was sampled.

    Gets:
        value: A sampled value for the yielded sample.

    Returns:
        values: The mean value of each sample.
            A numpy array with the same length as `population.
    """

    evaluations = np.empty((len(population), num_sample_evaluations))
    for i, sample in enumerate(population):
        for j in range(num_sample_evaluations):
            value = yield sample, step, mean
            if not isinstance(value, numbers.Real):
                raise ValueError("Must send a value for the yielded sample")
            evaluations[i, j] = value
    return np.mean(evaluations, axis=-1)


def _uncertainty_handling(
    population: np.ndarray,
    values: np.ndarray,
    num_sample_evaluations: int,
    num_reevaluations: int,
    scale_sample_evaluations: float,
    noise_tolerance: float,
    mean: np.ndarray,
    step: int,
) -> Generator[Tuple[np.ndarray, int, np.ndarray], float, int]:
    """Update the number of times to evaluate each sample.

    This implements the uncertaintyHandling procedure of Heidrich-Meisner and Igel.
    """
    # Line 2: Copy existing values
    reev_values = values.copy()
    # Line 1: Replaces first `num_reevaluations` values with new estimates
    reev_values[:num_reevaluations] = yield from _cma_es_collect_samples(
        population[:num_reevaluations], num_sample_evaluations, mean=mean, step=step
    )

    # Line 5: Compute rank changes
    #
    # Note: This calculation differs from the calculation described by
    # Heidrich-Meisner and Igel because I could not make sense of their explanation.
    #
    # Note that values and reev_values are both indexed by i
    # They say to first calculate both:
    #   rank_L         = rank(concat[values, values])
    #   rank_L^reeval  = rank(concat[values, reev_values])
    #
    # Then the rank change is given by:
    #   Δ_i = | rank_L^reeval(i) - rank_L(i) | - 1
    #
    # Where rank_L(i) and rank_L^reeval(i) are the
    # "ranks ... of each reevaluated individual x_i"
    #
    # However, there are two values for x_i in these lists: one from `values` twice or
    # one from `values` and one from `reev_values`, so rank(i) is ambiguous.
    # Also note that if we pick any consistent assignment of which original index
    # corresponds to to rank(i) then consider the the case where values = reev_values:
    # We get rank_L^reeval = rank_L so Δ_i = -1, which does not make sense and
    # contradicts the later treatment of Δ_i as non-negative.
    #
    # The authors claim that this apparently "cumbersome and overly complex" procedure
    # is necessary because it ensure that the calculated rank change is stable when
    # values at the same index of values and reev_values are swapped.
    # I was not able to obtain the property in practice through any variant of their
    # method as described.
    #
    # If instead we set:
    #   [rank_L, rank_L^reeval] = rank(concat[values, reev_values])
    #
    # then we _do_ get stability under swapping values[i] and reev_values[i].
    # Also, the rank change equation:
    #   Δ_i = | rank_L^reeval(i) - rank_L(i) | - 1
    # now makes sense and produces values that are >= 0
    # and equal 0 if values = reev_values.
    population_size = len(values)
    L_argmax = np.argsort(np.concatenate([values, reev_values]))
    L_ranks = np.argsort(L_argmax)
    orig_ranks = L_ranks[:population_size]
    reev_ranks = L_ranks[population_size:]
    rank_changes = np.abs(orig_ranks - reev_ranks) - 1

    # Line 6: Uncertainty Level
    orig_tolerated_rank_changes = _tolerated_rank_changes(
        orig_ranks - (orig_ranks > reev_ranks),
        population_size=population_size,
        noise_tolerance=noise_tolerance,
    )
    reev_tolerated_rank_changes = _tolerated_rank_changes(
        reev_ranks - (reev_ranks > orig_ranks),
        population_size=population_size,
        noise_tolerance=noise_tolerance,
    )
    excess_rank_changes = (
        2 * rank_changes - orig_tolerated_rank_changes - reev_tolerated_rank_changes
    )
    # s
    uncertainty_level = np.mean(excess_rank_changes[:num_reevaluations])

    if uncertainty_level > 0:
        # Round up to avoid being stuck at 1 if scale_sample_evaluations < 2
        num_sample_evaluations = math.ceil(
            num_sample_evaluations * scale_sample_evaluations
        )
    else:
        num_sample_evaluations = int(num_sample_evaluations / scale_sample_evaluations)

    # Line 8: Integrate reevaluations into values
    values[:num_reevaluations] = (
        values[:num_reevaluations] + reev_values[:num_reevaluations]
    ) / 2

    return max(num_sample_evaluations, 1)


def _tolerated_rank_changes(
    ranks: np.ndarray, population_size: int, noise_tolerance: float
) -> np.ndarray:
    # This function implements Δ_θ^lim(rank)
    return np.quantile(
        np.abs(np.arange(1, 2 * population_size) - ranks[:, None]),
        noise_tolerance / 2,
        axis=-1,
    )
