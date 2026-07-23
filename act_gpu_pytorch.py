"""
Adaptive Chirplet Transform (ACT) -- Multi-Channel, PyTorch
=============================================================

PyTorch implementation of the Adaptive Chirplet Transform, supporting
multi-channel signals and two dictionary-generation strategies:

    "hybrid" -- chirplets are generated on CPU (NumPy) then uploaded to GPU.
    "gpu"    -- chirplets are generated directly on GPU (PyTorch tensors).

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

import joblib
import numpy as np
import psutil
import scipy.optimize as optimize
import torch


class ACT:
    """Adaptive Chirplet Transform engine (PyTorch, multi-channel) backed by
    a precomputed chirplet dictionary."""

    def __init__(
        self,
        FS=256,
        length=3840,
        dict_addr="dict_cache_torch.p",
        tc_info=(0, 3840, 1),
        fc_info=(0.7, 15, 0.2),
        logDt_info=(-4, -1, 0.3),
        c_info=(-30, 30, 3),
        complex=False,
        dtype=torch.float32,
        mode="hybrid",  # "hybrid" | "gpu"
        force_regenerate=False,
        mute=False,
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
        dtype : torch.dtype
            Floating point dtype used for tensors (default torch.float32).
        mode : str
            Dictionary generation strategy: "hybrid" (CPU-generated, GPU-hosted)
            or "gpu" (GPU-generated).
        force_regenerate : bool
            If True, rebuild the dictionary even if a cache file exists.
        mute : bool
            If True, suppress informational print statements.
        """
        self.FS = FS
        self.length = length
        self.dict_addr = dict_addr
        self.tc_info = tc_info
        self.fc_info = fc_info
        self.logDt_info = logDt_info
        self.c_info = c_info
        self.complex = complex
        self.dtype = dtype
        self.mode = mode.lower()
        assert self.mode in ("hybrid", "gpu"), f"mode must be 'hybrid' or 'gpu', got '{mode}'"

        # Note: unified memory has no equivalent here -- PyTorch doesn't
        # support it outside of CuPy, so it's intentionally omitted.

        # ---------- Devices ----------
        self.cpu_device = torch.device("cpu")
        self.gpu_device = torch.device("cuda") if torch.cuda.is_available() else self.cpu_device
        self.device = self.gpu_device  # always compute on GPU (or CPU fallback)

        if not mute:
            print("\n========== ACT INIT ==========")
            print(f"Mode              : {self.mode}")
            print(f"Compute device    : {self.device}")
            print(f"CPU cores         : {psutil.cpu_count(logical=True)}")
            print(f"CUDA available    : {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device name  : {torch.cuda.get_device_name(0)}")
            print("================================\n")

        # ---------- Load / Generate Dictionary ----------
        if os.path.exists(self.dict_addr) and not force_regenerate:
            dict_np, param_np = joblib.load(self.dict_addr)
            if not mute:
                print("[DICT] Loaded dictionary from cache")
        else:
            if self.mode == "gpu":
                dict_np, param_np = self._generate_dictionary_gpu()
            else:
                dict_np, param_np = self._generate_dictionary_cpu()
            joblib.dump((dict_np, param_np), self.dict_addr)
            if not mute:
                print("[DICT] Dictionary cached")

        # Move dictionary to the compute device.
        self.dict_mat = torch.from_numpy(dict_np).to(self.device).to(self.dtype)
        self.param_mat = torch.from_numpy(param_np).to(self.device).to(self.dtype)

        if not mute:
            print(f"[DICT] dict_mat device : {self.dict_mat.device}")
            print(f"[DICT] param_mat device: {self.param_mat.device}\n")

    # ------------------------------------------------------------------ #
    # Chirplet generators (CPU / GPU)
    # ------------------------------------------------------------------ #
    def g_cpu(self, tc, fc, logDt, c):
        """
        Generate a chirplet on CPU (NumPy), normalized to unit norm.

        Parameters
        ----------
        tc : float
            Time center, in samples (converted internally to seconds).
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
        tc = float(tc)
        fc = float(fc)
        logDt = float(logDt)
        c = float(c)

        t = np.arange(self.length) / self.FS  # seconds
        tc_s = tc / self.FS  # convert samples -> seconds
        Dt = np.exp(logDt)

        gauss = np.exp(-0.5 * ((t - tc_s) / Dt) ** 2)
        phase = np.exp(2j * np.pi * (c * (t - tc_s) ** 2 + fc * (t - tc_s)))
        atom = gauss * phase

        if not self.complex:
            atom = atom.real

        norm = np.linalg.norm(atom)
        if norm > 0:
            atom /= norm

        return atom.astype(np.float32)

    def g_gpu(self, tc, fc, logDt, c):
        """
        Generate a chirplet directly on GPU (PyTorch), normalized to unit norm.

        Same parameters as `g_cpu`. Phase is computed in float64 for numerical
        precision before being cast back down to `self.dtype` (or complex64
        when `self.complex` is True).

        Returns
        -------
        torch.Tensor
            The normalized chirplet, shape (length,), on `self.device`.
        """
        t = torch.arange(self.length, device=self.device, dtype=self.dtype) / self.FS
        tc_s = float(tc) / self.FS  # convert samples -> seconds
        Dt = torch.exp(torch.tensor(float(logDt), device=self.device, dtype=self.dtype))
        dt = t - tc_s

        gauss = torch.exp(-0.5 * (dt / Dt) ** 2)

        # Compute phase in float64 for precision; use cos() to stay real when not complex.
        angle = 2 * torch.pi * (float(c) * dt.double() ** 2 + float(fc) * dt.double())
        if self.complex:
            phase = torch.exp(1j * angle).to(torch.complex64)
            atom = gauss.to(torch.complex64) * phase
        else:
            phase = torch.cos(angle).to(self.dtype)
            atom = gauss * phase

        norm = torch.linalg.norm(atom)
        if norm > 0:
            atom = atom / norm

        return atom

    def g(self, tc, fc, logDt, c):
        """
        Generate a chirplet and return it as a tensor on `self.device`.

        In "gpu" mode the chirplet is generated directly on GPU via `g_gpu`.
        In "hybrid" mode the chirplet is generated on CPU via `g_cpu` and then
        uploaded to `self.device`.
        """
        if self.mode == "gpu":
            return self.g_gpu(tc, fc, logDt, c)

        # hybrid: CPU generation, then upload.
        atom_np = self.g_cpu(tc, fc, logDt, c)
        return torch.from_numpy(atom_np).to(self.device)

    # ------------------------------------------------------------------ #
    # Optimization objective
    # ------------------------------------------------------------------ #
    def minimize_this(self, params, signal_np):
        """
        Objective for `scipy.optimize.minimize` (BFGS): negative absolute
        correlation between a candidate chirplet and the target signal.

        Uses the same chirplet generator as `self.g()` so hybrid and gpu modes
        remain internally consistent.

        Parameters
        ----------
        params : array-like
            [tc, fc, logDt, c] candidate chirplet parameters.
        signal_np : np.ndarray
            1D target signal (residue) for one channel.

        Returns
        -------
        float
            -abs(dot(chirplet, signal)); minimized to maximize correlation magnitude.
        """
        atom = self.g(*params)
        atom_np = atom.cpu().numpy() if isinstance(atom, torch.Tensor) else atom
        return -abs(np.dot(atom_np, signal_np))

    # ------------------------------------------------------------------ #
    # Dictionary generation
    # ------------------------------------------------------------------ #
    def _generate_dictionary_cpu(self):
        """
        Build the full chirplet dictionary on CPU (NumPy) by enumerating the
        Cartesian product of (tc, fc, logDt, c) parameter grids.

        Returns
        -------
        dict_np : np.ndarray, shape (dict_size, length)
        param_np : np.ndarray, shape (dict_size, 4)
        """
        tc_vals = np.arange(*self.tc_info)
        fc_vals = np.arange(*self.fc_info)
        logDt_vals = np.arange(*self.logDt_info)
        c_vals = np.arange(*self.c_info)

        K = len(tc_vals) * len(fc_vals) * len(logDt_vals) * len(c_vals)
        dict_np = np.zeros((K, self.length), dtype=np.float32)
        param_np = np.zeros((K, 4), dtype=np.float32)

        idx = 0
        for tc in tc_vals:
            for fc in fc_vals:
                for logDt in logDt_vals:
                    for c_val in c_vals:
                        dict_np[idx] = self.g_cpu(tc, fc, logDt, c_val)
                        param_np[idx] = [tc, fc, logDt, c_val]
                        idx += 1

        print("[DICT] Dictionary generated on CPU")
        return dict_np, param_np

    def _generate_dictionary_gpu(self):
        """
        Build the full chirplet dictionary directly on GPU (PyTorch) by
        enumerating the Cartesian product of (tc, fc, logDt, c) parameter
        grids, then transfer the result back to NumPy for caching.

        Returns
        -------
        dict_np : np.ndarray, shape (dict_size, length)
        param_np : np.ndarray, shape (dict_size, 4)
        """
        tc_vals = np.arange(*self.tc_info)
        fc_vals = np.arange(*self.fc_info)
        logDt_vals = np.arange(*self.logDt_info)
        c_vals = np.arange(*self.c_info)

        dict_list = []
        param_list = []
        for tc in tc_vals:
            for fc in fc_vals:
                for logDt in logDt_vals:
                    for c_val in c_vals:
                        atom = self.g_gpu(tc, fc, logDt, c_val)
                        dict_list.append(atom.cpu().numpy())
                        param_list.append([tc, fc, logDt, c_val])

        dict_np = np.stack(dict_list).astype(np.float32)
        param_np = np.array(param_list, dtype=np.float32)
        print("[DICT] Dictionary generated on GPU")
        return dict_np, param_np

    # ------------------------------------------------------------------ #
    # Dictionary search
    # ------------------------------------------------------------------ #
    def search_dictionary_batch(self, signals):
        """
        Find the best-matching chirplet for each channel in a batch.

        Parameters
        ----------
        signals : torch.Tensor
            Shape (channels, length), on `self.device`.

        Returns
        -------
        idx : torch.Tensor
            Index into the dictionary of the best-matching chirplet, per channel.
        val : torch.Tensor
            The corresponding projection (inner product) value, per channel.
        """
        proj = torch.matmul(self.dict_mat, signals.T)  # (dict_size, channels)
        idx = torch.argmax(torch.abs(proj), dim=0)  # (channels,)
        val = proj[idx, torch.arange(signals.shape[0], device=self.device)]
        return idx, val

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #
    def transform(self, signals, order=5, debug=False):
        """
        Compute the Adaptive Chirplet Transform (Matching Pursuit) for one or
        more channels.

        Parameters
        ----------
        signals : array-like or torch.Tensor, shape (L,) or (C, L)
            Input signal(s). A 1D input is treated as a single channel and
            the corresponding outputs are squeezed back to 1D.
        order : int
            Number of chirplets to extract per channel.
        debug : bool
            Currently unused; reserved for future progress logging.

        Returns
        -------
        dict
            Dictionary of NumPy arrays with keys:
                "params"      -- fitted chirp parameters, (order, C, 4) or (order, 4)
                "coeffs"      -- chirplet coefficients, (order, C) or (order,)
                "signal"      -- the original input signal(s)
                "approx"      -- reconstructed approximation of the signal(s)
                "residue"     -- final residual signal(s)
                "mse"         -- residual energy ratio per channel
                "norm_residue"-- residual norm ratio per channel
            If the input was 1D, the channel axis is squeezed out of every array.
        """
        # ---------- input normalisation ----------
        if not isinstance(signals, torch.Tensor):
            signals = torch.tensor(np.asarray(signals), dtype=self.dtype)
        signals = signals.to(self.device)

        squeezed = signals.ndim == 1
        if squeezed:
            signals = signals.unsqueeze(0)  # (1, L)

        C, L = signals.shape
        assert L == self.length, (
            f"Signal length {L} does not match dictionary length {self.length}"
        )

        approx = torch.zeros_like(signals, dtype=self.dtype)
        residue = signals.clone().to(self.dtype)
        param_list = torch.zeros((order, C, 4), dtype=self.dtype, device=self.device)
        coeff_list = torch.zeros((order, C), dtype=self.dtype, device=self.device)

        # ---------- Matching Pursuit ----------
        for p in range(order):
            inds, _ = self.search_dictionary_batch(residue)
            params0 = self.param_mat[inds]  # (C, 4) -- initial guess

            for c_idx in range(C):
                res_np = residue[c_idx].cpu().numpy()  # cache residue slice

                result = optimize.minimize(
                    self.minimize_this,
                    params0[c_idx].cpu().numpy(),
                    args=(res_np,),
                    method="BFGS",
                    options={"maxiter": 200, "gtol": 1e-5},
                )

                new_params = torch.tensor(result.x, dtype=self.dtype, device=self.device)
                atom = self.g(*new_params)

                # Guarantee real, float, same device.
                if atom.is_complex():
                    atom = atom.real
                atom = atom.to(self.dtype).to(self.device)

                coeff = torch.dot(atom, residue[c_idx])
                approx[c_idx] += atom * coeff
                residue[c_idx] -= atom * coeff

                param_list[p, c_idx] = new_params
                coeff_list[p, c_idx] = coeff

        # ---------- Safe error metrics ----------
        def _real_float(x):
            return x.real.float() if x.is_complex() else x.float()

        approx_f = _real_float(approx)
        residue_f = _real_float(residue)
        signals_f = _real_float(signals)

        sig_power = torch.sum(signals_f ** 2, dim=1)
        # Avoid division by zero.
        mse = torch.sum(residue_f ** 2, dim=1) / sig_power.clamp(min=1e-12)
        norm_residue = (
            torch.linalg.norm(residue_f, dim=1)
            / torch.linalg.norm(signals_f, dim=1).clamp(min=1e-12)
        )

        # ---------- Build output ----------
        out = {
            "params": param_list.cpu().numpy(),
            "coeffs": coeff_list.cpu().numpy(),
            "signal": signals_f.cpu().numpy(),
            "approx": approx_f.cpu().numpy(),
            "residue": residue_f.cpu().numpy(),
            "mse": mse.cpu().numpy(),
            "norm_residue": norm_residue.cpu().numpy(),
        }

        # Squeeze back out the channel axis if the input was 1D.
        # Each array has a different shape, so squeeze along the correct axis:
        #   signal/approx/residue : (1, L)        -> (L,)        axis=0
        #   mse/norm_residue      : (1,)           -> scalar      axis=0
        #   coeffs                : (order, 1)     -> (order,)    axis=1
        #   params                : (order, 1, 4)  -> (order, 4) axis=1
        if squeezed:
            def _sq(key, v):
                if key in ("signal", "approx", "residue", "mse", "norm_residue"):
                    return np.squeeze(v, axis=0)
                elif key == "coeffs":  # (order, 1) -> (order,)
                    return np.squeeze(v, axis=1)
                elif key == "params":  # (order, 1, 4) -> (order, 4)
                    return np.squeeze(v, axis=1)
                return v

            out = {k: _sq(k, v) for k, v in out.items()}

        return out