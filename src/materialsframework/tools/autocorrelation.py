r"""Autocorrelation and Green-Kubo integration utilities.

This module provides the autocorrelation and integration primitives used by
the Green-Kubo thermal conductivity workflow in
:mod:`materialsframework.analysis.green_kubo`:

- :func:`autocorrelate_tensor` — outer-product autocorrelation
  :math:`\langle x_\alpha(0)\, x_\beta(t)\rangle` of a vector time series,
  returning the full ``(T, D, D)`` tensor.
- :func:`autocorrelate` — trace (dot-product) autocorrelation
  :math:`\langle \mathbf{x}(0)\cdot\mathbf{x}(t)\rangle`.
- :func:`green_kubo_tensor` — full anisotropic thermal conductivity tensor
  :math:`\kappa_{\alpha\beta}` from a heat-flux time series.
- :func:`green_kubo_integral` — isotropic thermal conductivity
  :math:`\kappa = \frac{1}{3}\mathrm{tr}(\kappa_{\alpha\beta})`, derived from
  :func:`green_kubo_tensor`.

The formulation follows Langer et al., *Phys. Rev. B* **108**, L100302 (2023).

Mean subtraction
================
The autocorrelation routines subtract the empirical mean from the time
series by default (``subtract_mean=True``).  In equilibrium the heat flux
has :math:`\langle\mathbf{J}\rangle = 0` exactly, but on a finite
trajectory the empirical mean is non-zero by chance.  Without subtraction
that DC component contaminates the ACF as a constant
:math:`\langle\mathbf{J}\rangle\!\cdot\!\langle\mathbf{J}\rangle` offset
that integrates linearly in :math:`\tau` and prevents the running
:math:`\kappa(\tau)` from plateauing.  Subtraction removes this bias and
is the standard Green-Kubo practice.

Known limitation:
    No noise reduction on the HFACF beyond mean subtraction is applied —
    the current implementation uses a direct trapezoidal integral, so the
    running :math:`\kappa(\tau)` must still be visually inspected for a
    plateau. Cepstral-analysis denoising per Knoop, Scheffler and
    Carbogno, *Phys. Rev. B* **107**, 224304 (2023), is future work.
"""

from __future__ import annotations

import numpy as np

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

# Fundamental constants (SI)
_EV_TO_J: float = 1.602176634e-19  # J per eV
_AMU_TO_KG: float = 1.66053906660e-27  # kg per amu
_KB_J_PER_K: float = 1.380649e-23  # J/K

# Unit conversion: ASE flux [eV * sqrt(eV/amu) / A^3] -> SI [W / m^2]
_J_ASE_TO_SI: float = _EV_TO_J * np.sqrt(_EV_TO_J / _AMU_TO_KG) * 1e30  # ~1.574e15


def autocorrelate_tensor(
    x: np.ndarray,
    unbiased: bool = True,
    subtract_mean: bool = True,
) -> np.ndarray:
    r"""Outer-product autocorrelation of a vector time series.

    Computes the estimator

    .. math::

        R_{\alpha\beta}(t) = \langle x_\alpha(0)\, x_\beta(t) \rangle

    via the Wiener-Khinchin theorem with zero-padding to length ``2T`` to
    avoid circular wrap-around. The result is **always symmetrised**:
    :math:`\hat R_{\alpha\beta}` and :math:`\hat R_{\beta\alpha}` are two
    different finite-sample estimates of the same true correlation under
    time-reversal invariance of equilibrium ensembles, so

    .. math::

        \hat R_{\alpha\beta}^{\text{sym}}(t) =
            \tfrac{1}{2}\!\left[\hat R_{\alpha\beta}(t) + \hat R_{\beta\alpha}(t)\right]

    is the strictly better estimator (variance halved on the off-diagonals).

    Args:
        x (np.ndarray): Time series of shape ``(T, D)`` (``D`` is typically
            3 for a Cartesian vector field such as the heat flux).
        unbiased (bool, optional): If True (default), normalise each lag
            ``t`` by ``T - t`` (unbiased estimator —
            :math:`\mathbb{E}[\hat R(t)] = R(t)` exactly, but variance
            :math:`\propto 1/(T-t)` blows up at the tail). If False,
            normalise by ``T`` for every lag (biased estimator —
            :math:`\mathbb{E}[\hat R(t)] = (1 - t/T) R(t)` so the tail is
            damped to zero, which is the standard choice when the result
            will be integrated over time, as in Green-Kubo).
        subtract_mean (bool, optional): If True (default) subtract the
            empirical mean of ``x`` along the time axis before computing
            the autocorrelation. This is the standard Green-Kubo
            convention and removes a finite-sample DC bias that would
            otherwise contaminate the running integral with a linearly
            growing offset.

    Returns:
        np.ndarray: Autocorrelation tensor of shape ``(T, D, D)``,
        symmetrised in the trailing two axes. The diagonal component
        ``acf[t, alpha, alpha]`` is the standard autocorrelation of
        ``x[:, alpha]``; off-diagonal components capture cross-correlations
        between Cartesian axes.

    Raises:
        ValueError: If ``x`` is not 2-dimensional.
    """
    if x.ndim != 2:
        raise ValueError(
            f"autocorrelate_tensor expects a (T, D) array, got shape {x.shape}."
        )
    if subtract_mean:
        x = x - x.mean(axis=0, keepdims=True)
    n, d = x.shape
    ffts = [np.fft.rfft(x[:, alpha], n=2 * n) for alpha in range(d)]
    acf = np.empty((n, d, d))
    for alpha in range(d):
        for beta in range(d):
            acf[:, alpha, beta] = np.fft.irfft(np.conj(ffts[alpha]) * ffts[beta])[:n].real

    if unbiased:
        acf /= (n - np.arange(n))[:, None, None]
    else:
        acf /= n

    # Symmetrise alpha<->beta: under time-reversal invariance of equilibrium
    # ensembles R_{alpha,beta}(t) = R_{beta,alpha}(t), so averaging the two
    # finite-sample estimators is strictly variance-reducing.
    acf = 0.5 * (acf + acf.transpose(0, 2, 1))
    return acf


def autocorrelate(
    x: np.ndarray,
    unbiased: bool = True,
    subtract_mean: bool = True,
) -> np.ndarray:
    r"""Dot-product autocorrelation ``<x(0).x(t)>``.

    For a scalar series this is the usual :math:`\langle x(0) x(t)\rangle`.
    For a vector series ``(T, D)`` it returns the trace of
    :func:`autocorrelate_tensor`, i.e. :math:`\sum_\alpha \langle x_\alpha(0)
    x_\alpha(t)\rangle`.

    Args:
        x (np.ndarray): Time series of shape ``(T,)`` or ``(T, D)``.
        unbiased (bool, optional): If True (default) use the unbiased
            estimator (divide each lag by ``T - t``). If False use the
            biased estimator (divide every lag by ``T``); see
            :func:`autocorrelate_tensor` for the trade-off. Defaults to
            True.
        subtract_mean (bool, optional): If True (default) subtract the
            empirical mean before correlating. See
            :func:`autocorrelate_tensor` for rationale.

    Returns:
        np.ndarray: Autocorrelation of shape ``(T,)``. The value at lag 0
        equals :math:`\sum_\alpha \langle x_\alpha^2\rangle` averaged over
        the trajectory.
    """
    if x.ndim == 1:
        if subtract_mean:
            x = x - x.mean()
        n = len(x)
        f = np.fft.rfft(x, n=2 * n)
        acf = np.fft.irfft(f * np.conj(f))[:n].real
        if unbiased:
            acf /= n - np.arange(n)
        else:
            acf /= n
        return acf
    return np.trace(
        autocorrelate_tensor(x, unbiased=unbiased, subtract_mean=subtract_mean),
        axis1=1,
        axis2=2,
    )


def _cumulative_trapezoid(y: np.ndarray, dx: float, axis: int = 0) -> np.ndarray:
    """Cumulative trapezoidal integral along ``axis`` with a leading zero.

    Equivalent to :func:`numpy.cumulative_trapezoid(y, dx=dx, axis=axis,
    initial=0)` but implemented without the NumPy-2.0 dependency. The
    returned array has the same shape as ``y``; the first slice along
    ``axis`` is zero and the last slice equals the full trapezoidal
    integral.

    Args:
        y (np.ndarray): Integrand samples.
        dx (float): Spacing between samples along ``axis``.
        axis (int, optional): Axis along which to integrate. Defaults to 0.

    Returns:
        np.ndarray: Cumulative integral, same shape as ``y``.
    """
    y_moved = np.moveaxis(y, axis, 0)
    increments = 0.5 * (y_moved[:-1] + y_moved[1:]) * dx
    head = np.zeros_like(y_moved[:1])
    cum = np.concatenate([head, np.cumsum(increments, axis=0)], axis=0)
    return np.moveaxis(cum, 0, axis)


def green_kubo_tensor(
    heat_flux: np.ndarray,
    dt_fs: float,
    temperature: float,
    volume: float,
    unbiased: bool = True,
    t_cutoff_fs: float | None = None,
    subtract_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Full thermal-conductivity tensor from a heat-flux time series.

    Evaluates the anisotropic Green-Kubo relation

    .. math::

        \kappa_{\alpha\beta} = \frac{V}{k_B T^2}
            \int_0^{\tau_{\max}}
            \langle J_\alpha(0)\, J_\beta(t)\rangle\, dt

    using :func:`autocorrelate_tensor` (always symmetrised in
    :math:`\alpha\beta`) followed by cumulative trapezoidal integration.
    ``kappa_tensor == kappa_tensor_cumulative[-1]`` by construction.

    Args:
        heat_flux (np.ndarray): Heat-flux time series of shape ``(T, 3)`` in
            ASE internal units ``eV * sqrt(eV/amu) / A^3``.
        dt_fs (float): Time between consecutive flux samples in fs.
        temperature (float): Ensemble temperature in Kelvin.
        volume (float): Simulation cell volume in cubic angstroms.
        unbiased (bool, optional): Forwarded to :func:`autocorrelate_tensor`.
            Defaults to True (unbiased estimator). Set to False to use the
            biased estimator, which damps the high-lag noise to zero and
            usually gives a much better behaved running :math:`\kappa(\tau)`
            for finite trajectories.
        t_cutoff_fs (float | None, optional): If provided, truncate the
            HFACF and the cumulative integral at lag ``t_cutoff_fs`` (in
            fs). Useful to stop the random-walk noise of the unbiased
            estimator from polluting the result. Defaults to None
            (integrate over the full available range).
        subtract_mean (bool, optional): If True (default) subtract the
            empirical mean of the heat flux before correlating. Recommended
            for finite trajectories — otherwise a non-zero
            :math:`\langle J\rangle` adds a constant offset to the HFACF
            that makes the running integral diverge linearly.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - ``kappa_tensor`` — shape ``(3, 3)``, thermal-conductivity
              tensor in W/(m*K). Off-diagonal elements reflect anisotropy;
              for a cubic crystal the tensor should be diagonal with equal
              entries within statistical error.
            - ``kappa_tensor_cumulative`` — shape ``(T_eff, 3, 3)``, running
              tensor :math:`\kappa_{\alpha\beta}(\tau)` for convergence
              diagnostics; starts at zero. ``T_eff`` is the truncated length
              when ``t_cutoff_fs`` is set, otherwise ``T``.
    """
    acf_tensor = autocorrelate_tensor(
        heat_flux, unbiased=unbiased, subtract_mean=subtract_mean,
    )  # (T, 3, 3), ASE^2 units

    if t_cutoff_fs is not None:
        n_keep = min(int(np.ceil(t_cutoff_fs / dt_fs)) + 1, acf_tensor.shape[0])
        acf_tensor = acf_tensor[:n_keep]

    dt_s = dt_fs * 1e-15
    volume_m3 = volume * 1e-30

    acf_si = acf_tensor * _J_ASE_TO_SI**2  # (W/m^2)^2

    prefactor = volume_m3 / (_KB_J_PER_K * temperature**2)
    kappa_cum = prefactor * _cumulative_trapezoid(acf_si, dx=dt_s, axis=0)
    kappa = kappa_cum[-1]

    return kappa, kappa_cum


def green_kubo_integral(
    heat_flux: np.ndarray,
    dt_fs: float,
    temperature: float,
    volume: float,
    unbiased: bool = True,
    t_cutoff_fs: float | None = None,
    subtract_mean: bool = True,
) -> tuple[float, np.ndarray]:
    r"""Isotropic thermal conductivity from a heat-flux time series.

    Evaluates :math:`\kappa = \frac{1}{3} \mathrm{tr}(\kappa_{\alpha\beta})`
    by delegating to :func:`green_kubo_tensor` and tracing over Cartesian
    indices. This is the standard Green-Kubo expression

    .. math::

        \kappa = \frac{V}{3 k_B T^2} \int_0^{\tau_{\max}}
                   \langle \mathbf{J}(0) \cdot \mathbf{J}(t) \rangle\, dt,

    with ``kappa == kappa_cumulative[-1]`` by construction.

    Args:
        heat_flux (np.ndarray): Heat-flux time series of shape ``(T, 3)`` in
            ASE internal units ``eV * sqrt(eV/amu) / A^3``.
        dt_fs (float): Time between consecutive flux samples in fs.
        temperature (float): Ensemble temperature in Kelvin.
        volume (float): Simulation cell volume in cubic angstroms.
        unbiased (bool, optional): See :func:`green_kubo_tensor`. Defaults
            to True.
        t_cutoff_fs (float | None, optional): See :func:`green_kubo_tensor`.
            Defaults to None.
        subtract_mean (bool, optional): See :func:`green_kubo_tensor`.
            Defaults to True.

    Returns:
        tuple[float, np.ndarray]:
            - ``kappa`` — isotropic thermal conductivity in W/(m*K).
            - ``kappa_cumulative`` — running integral :math:`\kappa(\tau)`
              of shape ``(T_eff,)``; starts at zero.
    """
    kappa_tensor, kappa_tensor_cum = green_kubo_tensor(
        heat_flux, dt_fs, temperature, volume,
        unbiased=unbiased, t_cutoff_fs=t_cutoff_fs, subtract_mean=subtract_mean,
    )
    kappa = float(np.trace(kappa_tensor)) / 3.0
    kappa_cumulative = np.trace(kappa_tensor_cum, axis1=1, axis2=2) / 3.0
    return kappa, kappa_cumulative
