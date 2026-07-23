"""
Adaptive Chirplet Transform (ACT).

Implements a matching-pursuit-style decomposition of a 1-D signal onto a
dictionary of Gaussian chirplets. Each chirplet is parameterized by:

    tc     -- time center (in samples)
    fc     -- center frequency (Hz)
    logDt  -- log of the Gaussian window's time spread
    c      -- chirp rate

At each iteration, the chirplet with the highest correlation to the
current residual is selected and then locally refined with BFGS to maximize
that correlation. The refined chirplet is subtracted from the residual, and the
process repeats for a fixed number of chirplets (considered as the "order").
"""

import os
from typing import Optional, Tuple

import joblib
import numpy as np
import scipy.optimize as optimize

# Index of each parameter within a parameter vector / row of `param_mat`.
PARAM_TC, PARAM_FC, PARAM_LOGDT, PARAM_C = range(4)


class ACT:
    """Adaptive Chirplet Transform via matching pursuit.

    On construction, a family of chirplets is either loaded from
    a cache file or generated fresh (and then cached) from the ranges
    supplied for each parameter.
    """

    def __init__(
        self,
        FS: int = 256,
        length: int = 3840,
        dict_addr: str = "dict_cache.p",
        tc_info: Tuple[float, float, float] = (0, 3840, 1),
        fc_info: Tuple[float, float, float] = (0.7, 15, 0.2),
        logDt_info: Tuple[float, float, float] = (-4, -1, 0.3),
        c_info: Tuple[float, float, float] = (-30, 30, 3),
        complex: bool = False,
        force_regenerate: bool = False,
        mute: bool = False,
    ):
        """
        Args:
            FS: Sampling rate in Hz.
            length: Number of samples in each signal / chirplet.
            dict_addr: Path used to cache/load the generated dictionary.
            tc_info: (start, stop, step) passed to np.arange for time-center
                candidates, in samples.
            fc_info: (start, stop, step) passed to np.arange for center-
                frequency candidates, in Hz.
            logDt_info: (start, stop, step) passed to np.arange for log
                window-width candidates.
            c_info: (start, stop, step) passed to np.arange for chirp-rate
                candidates.
            complex: If True, keep chirplets complex-valued; otherwise use
                only their real part.
            force_regenerate: If True, rebuild the dictionary even if a
                cache file exists at `dict_addr`.
            mute: If True, suppress progress/status messages.
        """
        self.FS = FS
        self.length = length
        self.dict_addr = dict_addr

        self.tc_info = tc_info
        self.fc_info = fc_info
        self.logDt_info = logDt_info
        self.c_info = c_info

        self.complex = complex
        self.float32 = True

        if not mute:
            print("\n=== INITIALIZING ADAPTIVE CHIRPLET TRANSFORM MODULE ===\n")

        if os.path.exists(self.dict_addr) and not force_regenerate:
            if not mute:
                print("Found cached chirplet dictionary. Loading...")
            self.dict_mat, self.param_mat = joblib.load(self.dict_addr)
        else:
            if not mute:
                print("Generating chirplet dictionary...")
            self.generate_chirplet_dictionary(debug=False)
            if not mute:
                print("Caching dictionary...")
            joblib.dump((self.dict_mat, self.param_mat), self.dict_addr)
            if not mute:
                print("Done.")

        if not mute:
            print("=== DONE INITIALIZING ACT MODULE ===\n")

    def g(
        self,
        tc: float = 0,
        fc: float = 1,
        logDt: float = 0,
        c: float = 0,
    ) -> np.ndarray:
        """Generate a single Gaussian chirplet, normalized to unit energy.

        Args:
            tc: Time center, in samples.
            fc: Center frequency, in Hz.
            logDt: Log of the Gaussian window's time spread.
            c: Chirp rate.

        Returns:
            A 1-D array of length `self.length` containing the chirplet
            (real-valued unless `self.complex` is True), with unit L2 norm.
        """
        tc /= self.FS  # convert time center from samples to seconds
        Dt = np.exp(logDt)
        t = np.arange(self.length) / self.FS

        gaussian_window = np.exp(-0.5 * ((t - tc) / Dt) ** 2)
        complex_exp = np.exp(2j * np.pi * (c * (t - tc) ** 2 + fc * (t - tc)))

        chirplet = gaussian_window * complex_exp
        if not self.complex:
            chirplet = np.real(chirplet)

        # Unit-energy normalization.
        norm = np.linalg.norm(chirplet)
        if norm > 0:
            chirplet /= norm

        if self.float32:
            chirplet = chirplet.astype(np.float32)

        return chirplet

    def generate_chirplet_dictionary(
        self, debug: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the full chirplet dictionary from the configured parameter ranges.

        Populates and returns `self.dict_mat` (one chirplet per row) and
        `self.param_mat` (the (tc, fc, logDt, c) tuple used for each chirplet).

        Args:
            debug: Currently unused; reserved for verbose logging.

        Returns:
            Tuple of (dict_mat, param_mat).
        """
        tc_vals = np.arange(*self.tc_info)
        fc_vals = np.arange(*self.fc_info)
        logDt_vals = np.arange(*self.logDt_info)
        c_vals = np.arange(*self.c_info)

        dict_size = len(tc_vals) * len(fc_vals) * len(logDt_vals) * len(c_vals)

        print(f"Dictionary length: {dict_size}")

        dict_mat = np.zeros([dict_size, self.length], dtype=np.float32)
        param_mat = np.zeros([dict_size, 4], dtype=np.float32)

        cnt = 0
        for tc in tc_vals:
            for fc in fc_vals:
                for logDt in logDt_vals:
                    for c in c_vals:
                        dict_mat[cnt] = self.g(tc, fc, logDt, c)
                        param_mat[cnt] = [tc, fc, logDt, c]
                        cnt += 1

        self.dict_mat = dict_mat
        self.param_mat = param_mat
        return dict_mat, param_mat

    def search_dictionary(self, signal: np.ndarray) -> Tuple[int, float]:
        """Find the chirplet with maximum absolute correlation to `signal`.

        Args:
            signal: The (residual) signal to correlate against the dictionary.

        Returns:
            Tuple of (index of best-matching chirplet, its (signed) projection
            coefficient onto `signal`).
        """
        projections = self.dict_mat.dot(signal)
        ind = np.argmax(np.abs(projections))
        return ind, projections[ind]

    def minimize_this(self, coeffs: np.ndarray, signal: np.ndarray) -> float:
        """BFGS objective: negative absolute correlation of a chirplet with `signal`.

        Minimizing this maximizes |chirplet . signal|, i.e. the chirplet's
        correlation (positive or negative) with the target signal.

        Args:
            coeffs: Candidate (tc, fc, logDt, c) chirp parameters.
            signal: The (residual) signal to correlate against.

        Returns:
            Negative absolute dot product between the generated chirplet and
            `signal`.
        """
        atom = self.g(*coeffs)
        return -1.0 * abs(atom.dot(signal))

    def transform(self, signal: np.ndarray, order: int = 5, debug: bool = False) -> dict:
        """Decompose `signal` into `order` chirplets via matching pursuit.

        At each of `order` iterations:
          1. Find the chirplet best correlated with the residual.
          2. Refine that chirplet's parameters with BFGS to (locally) maximize
             correlation with the residual.
          3. Regenerate the refined, unit-energy chirplet.
          4. Project the residual onto the refined chirplet to get a coefficient.
          5. Subtract the scaled chirplet from the residual and add it to the
             running approximation.

        Args:
            signal: The 1-D input signal to decompose.
            order: Number of chirplets to extract.
            debug: If True, print per-chirplet progress and optimizer warnings.

        Returns:
            Dict with keys:
                params: (order, 4) array of refined chirplet parameters.
                coeffs: (order,) array of chirplet coefficients.
                signal: The original input signal.
                error: Sum of the final residual (signed, not absolute).
                residue: The final residual signal.
                approx: The reconstructed approximation of `signal`.
                mse: Residual energy / signal energy.
                norm_residue: Residual L2 norm / signal L2 norm.
        """
        param_list = np.zeros([order, 4], dtype=np.float32)
        coeff_list = np.zeros(order, dtype=np.float32)
        approx = np.zeros(len(signal), dtype=np.float32)
        residue = np.copy(signal)

        if debug:
            print(f"Beginning {order}-order ACT transform...")

        for p in range(order):
            if debug:
                print(f"Processing chirplet {p + 1}/{order}...")

            # 1) Find best matching chirplet in the dictionary.
            ind, _ = self.search_dictionary(residue)
            params = self.param_mat[ind]

            # 2) Refine its parameters with a local optimizer.
            res = optimize.minimize(
                self.minimize_this, params, args=(residue,), method="BFGS"
            )
            new_params = res.x
            if res.status != 0 and debug:
                print(f"Optimizer did not converge: {res.message}")

            # 3) Generate the refined, unit-energy chirplet.
            updated_chirp = self.g(*new_params)

            # 4) Compute this chirplet's coefficient against the residual.
            coeff = updated_chirp.dot(residue)

            # 5) Update the residual and running approximation.
            residue -= updated_chirp * coeff
            approx += updated_chirp * coeff

            # 6) Store this iteration's parameters and coefficient.
            param_list[p] = new_params
            coeff_list[p] = coeff

        mse = np.sum(residue ** 2) / np.sum(signal ** 2)
        norm_residual = np.linalg.norm(residue) / np.linalg.norm(signal)

        return {
            "params": param_list,
            "coeffs": coeff_list,
            "signal": signal,
            "error": np.sum(residue),
            "residue": residue,
            "approx": approx,
            "mse": mse,
            "norm_residue": norm_residual,
        }