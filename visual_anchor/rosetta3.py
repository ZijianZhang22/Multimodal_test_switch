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
        anchor_scale: Scales known-pair term in refinement only (default 1 = official).
            For weight-only control, use 1/(1+lambda_z), with no third pool.
        **kwargs: Passed unchanged to ``WassersteinProcrustes``.

    Attributes (besides the official ones):
        mean_z: Training mean ``[1, d_Y]`` of Z, or None without Z.
        z_counts: Matched X-Z rows per refinement step (for logging).
    """

    def __init__(self, lambda_z: float = 0.0, init_cache: dict | None = None,
                 anchor_scale: float = 1.0, lambda_y: float = 1.0,
                 init_mode: str = "rosetta", anchor_centering: str = "pool", **kwargs) -> None:
        super().__init__(**kwargs)
        self.init_cache = init_cache  # share one dict across methods to reuse the geometric initialization
        if lambda_z < 0:
            raise ValueError("lambda_z must be non-negative")
        self.lambda_z = float(lambda_z)
        if lambda_y < 0:
            raise ValueError("lambda_y must be non-negative")
        if init_mode not in ("rosetta", "supervised"):
            raise ValueError("init_mode must be rosetta or supervised")
        if anchor_centering not in ("paired", "source_pool", "pool"):
            raise ValueError("anchor_centering must be paired, source_pool, or pool")
        self.anchor_centering = anchor_centering
        self.lambda_y = float(lambda_y)
        self.init_mode = init_mode
        if not 0 < anchor_scale <= 1.0:
            raise ValueError("anchor_scale must be in (0, 1]")
        self.anchor_scale = float(anchor_scale)
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
        if self.lambda_y == 0 and self._z is None and paired_x is None:
            raise ValueError("lambda_y=0 and no Z requires known pairs")
        try:
            if self.init_mode == "supervised":
                return self._fit_from_anchors(samples_x, samples_y, paired_x, paired_y, seed)
            return super().fit(samples_x, samples_y, paired_x, paired_y, randomness)
        finally:
            self._z, self._rng_z = None, None  # do not keep the training pool on the object

    def _fit_from_anchors(self, samples_x, samples_y, paired_x, paired_y, seed):
        """Anchor-only initialization; no English unpaired matching in initialization.

        Uses X unpaired-pool mean and optionally the Y unpaired-pool mean
        for more stable low-budget centering. `source_pool` keeps the Y mean
        estimated only from the known anchors. `paired` reproduces the older
        implementation. For `pool`, even with lambda_y=0 the Y pool
        contributes its MEAN (but not its samples or pseudo-pairs).
        Upstream PCA also uses unpaired English text.
        """
        if paired_x is None or paired_y is None or paired_x.shape[0] == 0:
            raise ValueError("supervised initialization requires nonempty known pairs")
        self.mean_x, self.mean_y = self._anchor_means(
            samples_x.float(), samples_y.float(), paired_x.float(), paired_y.float())
        px = center_and_normalize(paired_x.float(), self.mean_x)
        py = center_and_normalize(paired_y.float(), self.mean_y)
        x = center_and_normalize(samples_x.float(), self.mean_x)
        # Do not even normalize the unpaired English pool if lambda_y=0.
        y = (center_and_normalize(samples_y.float(), self.mean_y)
             if self.lambda_y > 0 else None)
        weight = polar(px.T @ py)
        randomness = Randomness(seed)
        for _ in range(self.num_iterations):
            weight = self.refinement_step(x, y, weight, px, py, randomness)
        self.weight = weight
        return self

    def _anchor_means(self, samples_x, samples_y, paired_x, paired_y):
        if self.anchor_centering == "paired":
            mean_x = paired_x.mean(dim=0, keepdim=True)
            mean_y = paired_y.mean(dim=0, keepdim=True)
        else:
            mean_x = samples_x.mean(dim=0, keepdim=True)
            mean_y = (samples_y.mean(dim=0, keepdim=True)
                      if self.anchor_centering == "pool"
                      else paired_y.mean(dim=0, keepdim=True))
        return mean_x, mean_y

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
        # Keep the *exact* official path when no control is active.
        if (self._z is None and self.anchor_scale == 1.0 and self.lambda_y == 1.0
                and self.init_mode == "rosetta"):
            # Critical: the official pathway remains bitwise identical.
            return super().refinement_step(samples_x, samples_y, weight, paired_x, paired_y, randomness)
        if self.lambda_y == 0 and self._z is None:
            # A: pairs only, no pseudo-pairs at all.
            if paired_x is None:
                raise ValueError("No objective terms: require known pairs or at least one target pool")
            return weight
        batch_x = samples_x[randomness.subset(samples_x.shape[0], self.batch_size)]
        y_pair = None
        z_pair = None
        if self.lambda_y > 0:
            if samples_y is None:
                raise ValueError("Y pool required when lambda_y > 0")
            batch_y = samples_y[randomness.subset(samples_y.shape[0], self.batch_size)]
            rows_y, cols_y = batched_hungarian_matching(
                batch_x @ weight, batch_y, max(batch_x.shape[0], batch_y.shape[0]))
            y_pair = (batch_x[rows_y], batch_y[cols_y])
        if self._z is not None:
            batch_z = self._z[self._rng_z.subset(self._z.shape[0], self.batch_size)]
            rows_z, cols_z = batched_hungarian_matching(
                batch_x @ weight, batch_z, max(batch_x.shape[0], batch_z.shape[0]))
            z_pair = (batch_x[rows_z], batch_z[cols_z])
            self.z_counts.append(len(rows_z))
        # Anchor coefficient is normalized by the active pseudo-pair count.
        # With lambda_y=0 this is the *visual* count, not a phantom Y count.
        k = len(y_pair[0]) if y_pair is not None else len(z_pair[0])
        cross_covariance = torch.zeros_like(weight)
        if y_pair is not None:
            cross_covariance += (self.lambda_y * k / len(y_pair[0])) * (y_pair[0].T @ y_pair[1])
        if z_pair is not None:
            cross_covariance += (self.lambda_z * k / len(z_pair[0])) * (z_pair[0].T @ z_pair[1])
        if paired_x is not None:
            cross_covariance += ((self.anchor_scale * k / paired_x.shape[0]) * paired_x.T) @ paired_y
        return polar(cross_covariance)

    @torch.inference_mode()
    def fit_supervised(self, samples_x: Tensor, samples_y: Tensor, paired_x: Tensor, paired_y: Tensor):
        """Pairs-only baseline: orthogonal Procrustes on the known pairs, in Rosetta's own preprocessing.

        Centers source and target with the configured `anchor_centering`
        means, and unit-normalizes both. `pool` uses both unpaired pool means;
        `source_pool` does not use the English pool mean; `paired` is legacy.
        No geometric initialization or unpaired Hungarian matching is used.
        """
        if paired_x is None or paired_x.shape[0] == 0:
            raise ValueError("the supervised baseline needs at least one known pair")
        samples_x, samples_y = samples_x.float(), samples_y.float()
        self.mean_x, self.mean_y = self._anchor_means(
            samples_x.float(), samples_y.float(), paired_x.float(), paired_y.float())
        px = center_and_normalize(paired_x.float(), self.mean_x)
        py = center_and_normalize(paired_y.float(), self.mean_y)
        self.weight, self.mean_z, self.z_counts = polar(px.T @ py), None, []
        return self

    @torch.inference_mode()
    def transform_target(self, samples: Tensor, mean: Tensor) -> Tensor:
        """Center and normalize samples already in Y's coordinates with a given mean (e.g. an image mean)."""
        return center_and_normalize(samples.float(), mean)
