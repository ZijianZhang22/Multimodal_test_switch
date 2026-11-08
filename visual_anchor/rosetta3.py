"""Rosetta (Wasserstein Procrustes + geometric initialization) with an optional third target pool Z.

Built on the official implementation, pinned to dominik-schnaus/unpaired-rosetta@fdfad84ea488f29702a61a455be71622c4ff5d49.
The official source is NOT modified; this file only subclasses it.

v1 design (details in README_rosetta3.md):
  * Geometric initialization and readout are the official ones and use only X (low-resource) and Y (English).
  * Each refinement step additionally Hungarian-matches the batch of X W to a batch of Z and adds those pseudo-pairs
    to the same Procrustes problem:

        W = polar( X_m^T Y_m  +  lambda_z * (k/k_z) * X_mz^T Z_mz  +  (k/p) * X_hat^T Y_hat )

    k = matched X-Y rows, k_z = matched X-Z rows, p = known pairs. The first and last terms are exactly the official
    ones, so lambda_z scales the Z pseudo-pairs relative to the Y pseudo-pairs.
  * Z lives in Y's coordinates (e.g. images mapped by a frozen V_IE) and is centered with ITS OWN mean, like the
    official code centers every space with its own mean. This removes a constant modality offset before matching.
  * Z batches are drawn from a separate random stream, so lambda_z = 0 (or Z = None) reproduces the official
    WassersteinProcrustes bit for bit.
"""

from __future__ import annotations

import torch
from torch import Tensor

import hashlib

from unpaired_rosetta import GeometricInitialization, WassersteinProcrustes
from unpaired_rosetta.assignment import batched_hungarian_matching
from unpaired_rosetta.linalg import center_and_normalize, polar
from unpaired_rosetta.randomness import Randomness

Z_STREAM_OFFSET = 1_000_003  # seed offset of the independent random stream for Z


def _digest(tensor: Tensor | None) -> str | None:
    return None if tensor is None else hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


class WassersteinProcrustes3(WassersteinProcrustes):
    """Official Rosetta aligner plus an optional third target pool in Y's coordinates.

    Args:
        lambda_z: Weight of the X-Z pseudo-pairs relative to the X-Y pseudo-pairs. 0 = official Rosetta.
        init_cache: Optional dict shared across fits; reuses identical geometric initializations (bit-identical).
        **kwargs: Passed unchanged to ``WassersteinProcrustes``.

    Attributes (besides the official ones):
        mean_z: Training mean ``[1, d_Y]`` of Z, or None without Z.
        z_counts: Matched X-Z rows per refinement step (for logging).
    """

    def __init__(self, lambda_z: float = 0.0, init_cache: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.init_cache = init_cache  # share one dict across methods to reuse the geometric initialization
        if lambda_z < 0:
            raise ValueError("lambda_z must be non-negative")
        self.lambda_z = float(lambda_z)
        self.mean_z = None
        self.z_counts: list[int] = []
        self._z = None
        self._rng_z = None

    @torch.inference_mode()
    def fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None = None,
        paired_y: Tensor | None = None,
        randomness: Randomness | int = 0,
        samples_z: Tensor | None = None,
    ) -> WassersteinProcrustes3:
        """Same as the official ``fit``, with an optional third pool ``samples_z`` ``[m_z, d_Y]``."""
        seed = randomness.seed if isinstance(randomness, Randomness) else int(randomness)
        self.z_counts = []
        if samples_z is not None and self.lambda_z > 0:
            if samples_z.ndim != 2 or samples_z.shape[1] != samples_y.shape[1]:
                raise ValueError("samples_z must be [m_z, d_Y], in Y's coordinates")
            samples_z = samples_z.float()
            self.mean_z = samples_z.mean(dim=0, keepdim=True)
            self._z = center_and_normalize(samples_z, self.mean_z)
            self._rng_z = Randomness(seed + Z_STREAM_OFFSET)
        else:
            self.mean_z, self._z, self._rng_z = None, None, None
        try:
            return super().fit(samples_x, samples_y, paired_x, paired_y, randomness)
        finally:
            self._z, self._rng_z = None, None  # do not keep the training pool on the object

    def _fit(self, samples_x, samples_y, paired_x, paired_y, randomness):
        """Official ``_fit`` with an optional cache of the geometric initialization.

        The initialization uses only X, Y, the known pairs and the run's random streams, so every method with the same
        seed and pairs gets the same result. With ``self.init_cache`` (a dict) set, the first fit stores the averaged
        cross-covariance AND the random streams' state after it; later fits restore both, so results stay bit-identical
        to an uncached fit while skipping the expensive QAP restarts.
        """
        cache = getattr(self, "init_cache", None)
        if cache is None:
            return super()._fit(samples_x, samples_y, paired_x, paired_y, randomness)
        samples_x, samples_y = samples_x.float(), samples_y.float()
        self.mean_x = samples_x.mean(dim=0, keepdim=True)
        self.mean_y = samples_y.mean(dim=0, keepdim=True)
        samples_x = center_and_normalize(samples_x, self.mean_x)
        samples_y = center_and_normalize(samples_y, self.mean_y)
        if paired_x is not None and paired_x.shape[0] > 0:
            paired_x = center_and_normalize(paired_x.float(), self.mean_x)
            paired_y = center_and_normalize(paired_y.float(), self.mean_y)
        else:
            paired_x = paired_y = None
        key = (randomness.seed, self.num_clusters, self.num_restarts, self.batch_size,
               _digest(samples_x), _digest(samples_y), _digest(paired_x), _digest(paired_y))
        if key in cache:
            correspondence, states = cache[key]
            randomness.torch.set_state(states[0].clone())
            randomness.numpy.set_state(states[1])
            randomness.python.setstate(states[2])
        else:
            initialization = GeometricInitialization(
                self.num_clusters, self.num_restarts, self.batch_size, self.num_workers
            )
            correspondence = initialization(samples_x, samples_y, paired_x, paired_y, randomness)
            states = (randomness.torch.get_state().clone(), randomness.numpy.get_state(), randomness.python.getstate())
            cache[key] = (correspondence, states)
        weight = self.readout(samples_x, samples_y, correspondence, paired_x, paired_y, randomness)
        for _ in range(self.num_iterations):
            weight = self.refinement_step(samples_x, samples_y, weight, paired_x, paired_y, randomness)
        self.weight = weight
        return self

    def refinement_step(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        weight: Tensor,
        paired_x: Tensor | None,
        paired_y: Tensor | None,
        randomness: Randomness,
    ) -> Tensor:
        if self._z is None:
            return super().refinement_step(samples_x, samples_y, weight, paired_x, paired_y, randomness)
        # Lines 7-9 of Algorithm 1, copied from the official code: same draws from the same stream, in the same order.
        batch_x = samples_x[randomness.subset(samples_x.shape[0], self.batch_size)]
        batch_y = samples_y[randomness.subset(samples_y.shape[0], self.batch_size)]
        block_size = max(batch_x.shape[0], batch_y.shape[0])
        rows, columns = batched_hungarian_matching(batch_x @ weight, batch_y, block_size)
        matched = rows.shape[0]
        cross_covariance = batch_x[rows].T @ batch_y[columns]
        if paired_x is not None:
            cross_covariance = cross_covariance + ((matched / paired_x.shape[0]) * paired_x.T) @ paired_y
        # Extra term: the same batch of X W matched to a batch of Z, drawn from the independent stream.
        batch_z = self._z[self._rng_z.subset(self._z.shape[0], self.batch_size)]
        rows_z, columns_z = batched_hungarian_matching(
            batch_x @ weight, batch_z, max(batch_x.shape[0], batch_z.shape[0])
        )
        self.z_counts.append(int(rows_z.shape[0]))
        z_weight = self.lambda_z * matched / rows_z.shape[0]
        cross_covariance = cross_covariance + (z_weight * batch_x[rows_z].T) @ batch_z[columns_z]
        return polar(cross_covariance)

    @torch.inference_mode()
    def transform_target(self, samples: Tensor, mean: Tensor) -> Tensor:
        """Center and normalize samples already in Y's coordinates with a given mean (e.g. an image mean)."""
        return center_and_normalize(samples.float(), mean)
