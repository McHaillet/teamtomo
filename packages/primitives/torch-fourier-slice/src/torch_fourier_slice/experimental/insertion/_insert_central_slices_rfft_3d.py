"""Experimental Mojo-backed central-slice insertion (rfft layer).

The rfft-level adjoint operator: scatter 2D central slices into a 3D rfft volume
(Hermitian, DC at origin), optionally accumulating per-pixel weights for a
Wiener-style normalization. This is the experimental analogue of
:func:`torch_fourier_slice.insert_central_slices_rfft_3d` and is differentiable
w.r.t. the slices, weights, rotation_matrices, and 2D / 3D shifts.

Two rank forms share one Mojo kernel; the Python layer only squeezes / transposes:

- single volume: ``image_rfft (bp, h, w)`` -> ``volume (d, h, w)``
- multi-channel:  ``image_rfft (bp, bv, h, w)`` -> ``volumes (bv, d, h, w)``
  (poses shared across volumes, or per-volume via
  ``rotation_matrices (bv, bp, 3, 3)``).

``rotation_matrices``/``shifts_3d``/``shifts_2d`` and the Ewald arguments follow
the extraction module; the insertion applies the *conjugate* shift phase ramps
(the forward adjoint).
``weights`` is an optional real per-pixel tensor matching ``image_rfft``,
accumulated into a separate weight volume (returned alongside the data volume).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._backend._common import reconstruction_volume_shape
from .._backend._slice_3d import InsertSlices3D
from .._conventions import ewald_coefficient, to_zyx_matrices

if TYPE_CHECKING:
    import torch


def _insert(
    image_rfft,
    weights,
    rotation_matrices,
    shifts_3d,
    shifts_2d,
    oversampling,
    fftfreq_max,
    zyx_matrices,
    interpolation,
    apply_ewald_curvature,
    ewald_voltage_kv,
    ewald_flip_sign,
    ewald_px_size,
):
    """Run the differentiable kernel in its canonical ``(bv, bp, ...)`` layout."""
    volume_sidelength = reconstruction_volume_shape(image_rfft.shape[-2], oversampling)[
        1
    ]
    return InsertSlices3D.apply(
        image_rfft,
        weights,
        to_zyx_matrices(rotation_matrices, zyx_matrices),
        shifts_2d,
        shifts_3d,
        oversampling,
        fftfreq_max,
        interpolation,
        ewald_coefficient(
            volume_sidelength,
            apply_ewald_curvature,
            ewald_voltage_kv,
            ewald_flip_sign,
            ewald_px_size,
        ),
    )


def insert_central_slices_rfft_3d(
    image_rfft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
    interpolation: str = "linear",
    apply_ewald_curvature: bool = False,
    ewald_voltage_kv: float = 300.0,
    ewald_flip_sign: bool = False,
    ewald_px_size: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 2D central slices into one 3D rfft volume (Mojo scatter kernel).

    ``image_rfft`` is ``(bp, h, w)`` complex rfft slices (DC at origin); its device
    selects the backend. See the module docstring for the pose / weight parameters.

    Returns ``(volume, weight_volume)`` -- complex ``(d, h, w)`` accumulated data
    and real ``(d, h, w)`` accumulated weights (``None`` if ``weights`` is ``None``)
    -- on the input device.
    """
    if image_rfft.dim() != 3:
        raise ValueError(
            "image_rfft must be (bp, h, w) for a single volume; use "
            "insert_central_slices_rfft_3d_multichannel for (bp, bv, h, w)"
        )
    data, weight_vol = _insert(
        image_rfft,
        weights,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
        oversampling,
        fftfreq_max,
        zyx_matrices,
        interpolation,
        apply_ewald_curvature,
        ewald_voltage_kv,
        ewald_flip_sign,
        ewald_px_size,
    )
    if weights is None:
        return data.squeeze(0), None
    return data.squeeze(0), weight_vol.squeeze(0)


def insert_central_slices_rfft_3d_multichannel(
    image_rfft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
    interpolation: str = "linear",
    apply_ewald_curvature: bool = False,
    ewald_voltage_kv: float = 300.0,
    ewald_flip_sign: bool = False,
    ewald_px_size: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 2D central slices into a batch of 3D rfft volumes (Mojo kernel).

    ``image_rfft`` is ``(bp, bv, h, w)`` (pose-major); ``weights`` (if given)
    matches it. Poses are shared (``rotation_matrices (bp, 3, 3)``) or per-volume
    (``(bv, bp, 3, 3)``). See the module docstring for the shared parameters.

    Returns ``(volumes, weight_volumes)`` -- complex ``(bv, d, h, w)`` data and
    real ``(bv, d, h, w)`` weights (``None`` if ``weights`` is ``None``) -- on the
    input device.
    """
    if image_rfft.dim() != 4:
        raise ValueError("image_rfft must be (bp, bv, h, w) for multi-channel")
    imgs = image_rfft.transpose(0, 1).contiguous()  # (bp, bv, ...) -> (bv, bp, ...)
    w = weights.transpose(0, 1).contiguous() if weights is not None else None
    data, weight_vol = _insert(
        imgs,
        w,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
        oversampling,
        fftfreq_max,
        zyx_matrices,
        interpolation,
        apply_ewald_curvature,
        ewald_voltage_kv,
        ewald_flip_sign,
        ewald_px_size,
    )
    return data, (weight_vol if weights is not None else None)
