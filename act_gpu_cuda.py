"""
Adaptive Chirplet Transform (ACT)
=================================

CUDA implementation of the Adaptive Chirplet Transform, supporting
multi-channel signals and two dictionary-generation strategies:

    "hybrid = True"  -- chirplets are generated on CPU (NumPy) then uploaded to GPU.
    "hybrid = False" -- chirplets are generated directly on GPU (PyTorch tensors).

Regardless of mode, all computation during `transform` (dictionary search,
residue tracking) happens on `self.device` (GPU if available, else CPU).

The transform decomposes a signal into a sparse sum of Gaussian-windowed
chirplets, each parameterized by:

    tc      -- time center (samples)
    fc      -- center frequency (Hz)
    logDt   -- log of the Gaussian window duration
    c       -- chirp rate

A dictionary of candidate chirplets is generated (or loaded from a cache file),
and a matching-pursuit-style greedy search + local optimization is used to
fit each signal.
"""

import os

import cupy as cp
import joblib
import numpy as np
import psutil
import scipy.optimize as optimize
from cupy.cuda import memory


class ACT:
    """Adaptive Chirplet Transform engine backed by a precomputed chirplet dictionary."""

    def __init__(
        self,
        FS=256,
        length=3840,
        dict_addr="dict_cache.p",
        tc_info=(0, 3840, 1),
        fc_info=(0.7, 15, 0.2),
        logDt_info=(-4, -1, 0.3),
        c_info=(-30, 30, 3),
        complex=False,
        force_regenerate=False,
        mute=False,
        hybrid=True,
        unified_memory=False,
    ):
        """
        Parameters
        ----------
        FS : int
            Sampling frequency in Hz.
        length : int
            Signal length in samples.
        dict_addr : str
            Path to the cached dictionary file (loaded if present, else generated).
        tc_info, fc_info, logDt_info, c_info : tuple
            (start, stop, step) ranges used to build the parameter grid for
            time center, center frequency, log duration, and chirp rate.
        complex : bool
            If True, keep chirplets complex-valued; otherwise use the real part.
        force_regenerate : bool
            If True, rebuild the dictionary even if a cache file exists.
        mute : bool
            If True, suppress informational print statements.
        hybrid : bool
            If True, generate the dictionary on CPU (NumPy) then transfer to GPU.
            If False, generate directly on GPU (CuPy).
        unified_memory : bool
            If True, use CUDA managed (unified) memory for CuPy allocations.
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
        self.mute = mute
        self.hybrid = hybrid
        self.unified_memory = unified_memory

        # Enable unified memory globally if requested.
        if self.unified_memory:
            cp.cuda.set_allocator(memory.malloc_managed)

        self.device_info()

        # Load dictionary from cache, or generate and cache it.
        if os.path.exists(self.dict_addr) and not force_regenerate:
            if not mute:
                print("Found Chirplet Dictionary, Loading File...")
            dict_mat_np, param_mat_np = joblib.load(self.dict_addr)
            self.dict_mat = cp.asarray(dict_mat_np)
            self.param_mat = cp.asarray(param_mat_np)
        else:
            if not mute:
                print("Generating Chirplet Dictionary...")
            self.generate_chirplet_dictionary(debug=True)
            joblib.dump((self.dict_mat.get(), self.param_mat.get()), self.dict_addr)
            if not mute:
                print("Dictionary Cached.")

    # ------------------------------------------------------------------ #
    # Device info
    # ------------------------------------------------------------------ #
    def device_info(self):
        """Print basic CPU/GPU environment info for diagnostics."""
        print(f"CPU cores: {psutil.cpu_count(logical=True)}")
        try:
            print(f"GPU Devices detected: {cp.cuda.runtime.getDeviceCount()}")
        except Exception:
            print("No GPU detected")
        print(f"Hybrid mode: {self.hybrid}")
        print(f"Unified memory: {self.unified_memory}")

    # ------------------------------------------------------------------ #
    # Chirplets generators (CPU / GPU)
    # ------------------------------------------------------------------ #
    def g_cpu(self, tc=0, fc=1, logDt=0, c=0):
        """
        Generate a single chirplet on CPU (NumPy), normalized to unit norm.

        Parameters
        ----------
        tc : float
            Time center, in samples.
        fc : float
            Center frequency, in Hz.
        logDt : float
            Log of the Gaussian window's duration.
        c : float
            Chirp rate.

        Returns
        -------
        np.ndarray
            The normalized chirplet, shape (length,), dtype float32.
        """
        t = np.arange(self.length) / self.FS
        Dt = np.exp(logDt)
        gaussian_window = np.exp(-0.5 * ((t - tc / self.FS) / Dt) ** 2)
        complex_exp = np.exp(
            2j * np.pi * (c * (t - tc / self.FS) ** 2 + fc * (t - tc / self.FS))
        )
        atom = gaussian_window * complex_exp

        if not self.complex:
            atom = np.real(atom)

        norm = np.linalg.norm(atom)
        if norm > 0:
            atom /= norm

        return atom.astype(np.float32)

    def g(self, tc=0, fc=1, logDt=0, c=0):
        """
        Generate a single chirplet on GPU (CuPy), normalized to unit norm.

        Same parameters and behavior as `g_cpu`, but computed with CuPy arrays.
        """
        t = cp.arange(self.length) / self.FS
        Dt = cp.exp(logDt)
        gaussian_window = cp.exp(-0.5 * ((t - tc / self.FS) / Dt) ** 2)
        complex_exp = cp.exp(
            2j * cp.pi * (c * (t - tc / self.FS) ** 2 + fc * (t - tc / self.FS))
        )
        atom = gaussian_window * complex_exp

        if not self.complex:
            atom = cp.real(atom)

        norm = cp.linalg.norm(atom)
        if norm > 0:
            atom /= norm

        return atom.astype(cp.float32)

    # ------------------------------------------------------------------ #
    # Dictionary generation
    # ------------------------------------------------------------------ #
    def generate_chirplet_dictionary(self, debug=False):
        """
        Build the full chirplet dictionary by enumerating the Cartesian
        product of (tc, fc, logDt, c) parameter grids.

        Populates `self.dict_mat` (dict_size x length) and
        `self.param_mat` (dict_size x 4 parameters), either on CPU-then-GPU
        ("hybrid" mode) or directly on GPU.
        """
        tc_vals = np.arange(self.tc_info[0], self.tc_info[1], self.tc_info[2])
        fc_vals = np.arange(self.fc_info[0], self.fc_info[1], self.fc_info[2])
        logDt_vals = np.arange(self.logDt_info[0], self.logDt_info[1], self.logDt_info[2])
        c_vals = np.arange(self.c_info[0], self.c_info[1], self.c_info[2])

        dict_size = len(tc_vals) * len(fc_vals) * len(logDt_vals) * len(c_vals)
        print("Dictionary length:", dict_size)

        if self.hybrid:
            self._generate_dictionary_cpu(tc_vals, fc_vals, logDt_vals, c_vals, dict_size, debug)
        else:
            self._generate_dictionary_gpu(tc_vals, fc_vals, logDt_vals, c_vals, dict_size, debug)

        if debug:
            print("Dictionary Generated.")

    def _generate_dictionary_cpu(self, tc_vals, fc_vals, logDt_vals, c_vals, dict_size, debug):
        """Build the dictionary on CPU (NumPy), then transfer the result to GPU."""
        if debug:
            print("Generating dictionary on CPU...")

        dict_mat_np = np.zeros((dict_size, self.length), dtype=np.float32)
        param_mat_np = np.zeros((dict_size, 4), dtype=np.float32)

        cnt = 0
        for tc in tc_vals:
            for fc in fc_vals:
                for logDt in logDt_vals:
                    for c_val in c_vals:
                        dict_mat_np[cnt] = self.g_cpu(tc, fc, logDt, c_val)
                        param_mat_np[cnt] = np.array([tc, fc, logDt, c_val], dtype=np.float32)
                        cnt += 1

        self.dict_mat = cp.asarray(dict_mat_np)
        self.param_mat = cp.asarray(param_mat_np)

    def _generate_dictionary_gpu(self, tc_vals, fc_vals, logDt_vals, c_vals, dict_size, debug):
        """Build the dictionary directly on GPU (CuPy)."""
        if debug:
            print("Generating dictionary on GPU...")

        dict_mat = cp.zeros((dict_size, self.length), dtype=cp.float32)
        param_mat = cp.zeros((dict_size, 4), dtype=cp.float32)

        cnt = 0
        for tc in tc_vals:
            for fc in fc_vals:
                for logDt in logDt_vals:
                    for c_val in c_vals:
                        atom = self.g(tc, fc, logDt, c_val)
                        dict_mat[cnt] = atom
                        param_mat[cnt] = cp.array([tc, fc, logDt, c_val], dtype=cp.float32)
                        cnt += 1

        self.dict_mat = dict_mat
        self.param_mat = param_mat

    # ------------------------------------------------------------------ #
    # Input normalization
    # ------------------------------------------------------------------ #
    def _ensure_2d(self, signals):
        """
        Normalize `signals` into a 2D CuPy array of shape (batch, length).

        Accepts a single 1D signal, a list/tuple of signals, or an
        already-batched array (NumPy or CuPy), and validates signal length.
        """
        if isinstance(signals, np.ndarray):
            signals = cp.asarray(signals)

        if isinstance(signals, (list, tuple)):
            signals = cp.stack(
                [cp.asarray(s) if isinstance(s, np.ndarray) else s for s in signals],
                axis=0,
            )
        elif signals.ndim == 1:
            signals = signals[None, :]

        if signals.shape[1] != self.length:
            raise ValueError("Signal length mismatch")

        return signals

    # ------------------------------------------------------------------ #
    # Dictionary search
    # ------------------------------------------------------------------ #
    def search_dictionary_batch(self, signals):
        """
        Find the best-matching chirplet for each signal in a batch.

        Parameters
        ----------
        signals : cp.ndarray
            Shape (batch, length).

        Returns
        -------
        best_idx : cp.ndarray
            Index into the dictionary of the best-matching chirplet, per signal.
        best_val : cp.ndarray
            The corresponding projection (inner product) value, per signal.
        """
        projections = self.dict_mat.dot(signals.T)  # shape: (dict_size, batch)
        best_idx = cp.argmax(cp.abs(projections), axis=0)
        best_val = projections[best_idx, cp.arange(signals.shape[0])]
        return best_idx, best_val

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #
    def transform(self, signal, order=5, debug=False):
        """
        Decompose one or more signals into `order` chirplets via a
        greedy matching-pursuit search followed by local refinement.

        Parameters
        ----------
        signal : array-like
            A single signal (1D) or a batch of signals (2D / list of 1D arrays).
        order : int
            Number of chirplets to extract per signal.
        debug : bool
            If True, print progress for each chirplet extracted.

        Returns
        -------
        dict
            Dictionary with keys:
                "params"      -- (order, batch, 4) fitted chirplet parameters
                "coeffs"      -- (order, batch) chirplet coefficients
                "signal"      -- the original input signal(s)
                "error"       -- sum of residual (not squared) per signal
                "residue"     -- final residual signal(s)
                "approx"      -- reconstructed approximation of the signal(s)
                "mse"         -- residual energy ratio per signal
                "norm_residue"-- residual norm ratio per signal
        """
        signals = self._ensure_2d(signal)
        B, L = signals.shape

        param_list = cp.zeros((order, B, 4), dtype=cp.float32)
        coeff_list = cp.zeros((order, B), dtype=cp.float32)
        residue = cp.copy(signals)
        approx = cp.zeros_like(signals)

        if debug:
            print(f"Beginning {order}-order transform on {B} signal(s)")

        for p in range(order):
            if debug:
                print(f"Transforming Chirplet {p + 1}/{order}")

            # Greedy step: find the best dictionary chirplet for each signal's residue.
            inds, _ = self.search_dictionary_batch(residue)
            params_batch = self.param_mat[inds]

            # Refinement step: locally optimize each chirplet's parameters.
            for b in range(B):
                init_params = params_batch[b].get()
                res = optimize.minimize(
                    self.minimize_this,
                    init_params,
                    args=(residue[b].get(),),
                )
                new_params = cp.array(res.x)
                atom = self.g(*new_params)
                coeff = cp.dot(atom, residue[b])

                residue[b] -= atom * coeff
                approx[b] += atom * coeff
                param_list[p, b] = new_params
                coeff_list[p, b] = coeff

        mse = cp.sum(residue ** 2, axis=1) / cp.sum(signals ** 2, axis=1)
        norm_residue = cp.linalg.norm(residue, axis=1) / cp.linalg.norm(signals, axis=1)

        return {
            "params": param_list,
            "coeffs": coeff_list,
            "signal": signal,
            "error": cp.sum(residue, axis=1),
            "residue": residue,
            "approx": approx,
            "mse": mse,
            "norm_residue": norm_residue,
        }

    # ------------------------------------------------------------------ #
    # Optimization objective
    # ------------------------------------------------------------------ #
    def minimize_this(self, coeffs, signal):
        """
        Objective for `scipy.optimize.minimize`: negative absolute correlation
        between a candidate chirplet and the target signal (CPU-side).

        Parameters
        ----------
        coeffs : array-like
            [tc, fc, logDt, c] candidate chirplet parameters.
        signal : np.ndarray
            Target signal (residue) to match against.

        Returns
        -------
        float
            -abs(dot(chirplet, signal)); minimized to maximize correlation magnitude.
        """
        atom = self.g_cpu(tc=coeffs[0], fc=coeffs[1], logDt=coeffs[2], c=coeffs[3])
        return -1 * abs(np.dot(atom, signal))