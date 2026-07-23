"""
ACT_LEM: Adaptive Chirplet Transform via a Latin-hypercube / coarse-to-fine
matching pursuit search, accelerated on GPU with CuPy.

Given a 1D signal, ``ACT_LEM`` iteratively finds the best-fitting chirplets
(parameterized by time center, frequency center, log-duration, and
chirp rate) that explain the signal, using a matching-pursuit style
decomposition. Candidate chirplets are drawn from a dictionary (i.e. family
of chirplets) located with a Latin hypercube search followed by successive
coarse-to-fine grid refinement, then locally polished with a
gradient-based optimizer. Throughout this module, "dictionary" and "family
of chirplets" refer to the same underlying collection and are used
interchangeably.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cupy as cp
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import optimize
from scipy.stats import qmc


def _params_to_key(tc_r: Sequence[float], fc_r: Sequence[float],
                    logDt_r: Sequence[float], c_r: Sequence[float],
                    length: int, FS: int) -> str:
    """Build a stable cache key (SHA-1 hex digest) for a dictionary range spec."""
    keyobj = {"tc": tc_r, "fc": fc_r, "logDt": logDt_r, "c": c_r,
              "length": int(length), "FS": int(FS)}
    s = json.dumps(keyobj, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()


class ACT_LEM:
    """Adaptive Chirplet Transform, Latin-hypercube / coarse-to-fine search.

    Parameters
    ----------
    FS : int
        Sampling rate (Hz).
    length : int
        Number of samples in the signal being decomposed.
    tc_info, fc_info, logDt_info, c_info : Any
        Reserved metadata about each chirplet parameter (kept for API
        compatibility with callers; not consumed internally).
    float32 : bool
        If True, generated chirplets are cast to float32 (recommended for GPU
        throughput). If False, ``g()`` returns the full complex chirplet.
    mute : bool
        If True, suppresses informational progress prints (level markers,
        candidate summaries). Detailed per-sample debug prints are
        controlled separately by ``self.debug``.
    cache : bool
        If True, dictionary tensors (i.e. the family of chirplets) built by
        ``generate_dictionary_from_ranges`` are cached to disk (keyed by
        their parameter ranges) and reused.
    """

    def __init__(self, FS: int, length: int,
                 tc_info: Any, fc_info: Any, logDt_info: Any, c_info: Any,
                 float32: bool = True, mute: bool = False, cache: bool = False):
        self.FS = FS
        self.length = length
        self.tc_info = tc_info
        self.fc_info = fc_info
        self.logDt_info = logDt_info
        self.c_info = c_info
        self.float32 = float32
        self.mute = mute
        self.cache = cache
        self.debug = False

    # -------------------------------
    # Chirplet generators
    # -------------------------------
    def g(self, tc: float = 0, fc: float = 1, logDt: float = 0, c: float = 0) -> cp.ndarray:
        """Generate a single chirplet on the GPU.

        Returns the real part (float32) if ``self.float32`` is True,
        otherwise the full complex-valued chirplet.
        """
        tc = tc / self.FS
        Dt = cp.exp(logDt)
        t = cp.arange(self.length) / self.FS
        gaussian_window = cp.exp(-0.5 * ((t - tc) / Dt) ** 2)
        complex_exp = cp.exp(2j * cp.pi * (c * (t - tc) ** 2 + fc * (t - tc)))
        chirplet = gaussian_window * complex_exp
        if self.float32:
            return cp.real(chirplet).astype(cp.float32)
        else:
            return chirplet

    def g_cpu(self, tc: float = 0, fc: float = 1, logDt: float = 0, c: float = 0) -> np.ndarray:
        """CPU (NumPy) equivalent of ``g()``, used where GPU transfer isn't worth it."""
        tc = tc / self.FS
        Dt = np.exp(logDt)
        t = np.arange(self.length) / self.FS
        gaussian_window = np.exp(-0.5 * ((t - tc) / Dt) ** 2)
        complex_exp = np.exp(2j * np.pi * (c * (t - tc) ** 2 + fc * (t - tc)))
        chirplet = gaussian_window * complex_exp
        return np.real(chirplet).astype(np.float32)

    # -------------------------------
    # Dictionary generation
    # -------------------------------
    def generate_dictionary_from_ranges(
        self,
        tc_range: Tuple[float, float, float],
        fc_range: Tuple[float, float, float],
        logDt_range: Tuple[float, float, float],
        c_range: Tuple[float, float, float],
        cache: Optional[bool] = None,
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Build (or load from cache) a dictionary -- i.e. family of
        chirplets -- over a parameter grid.

        Each of ``tc_range``, ``fc_range``, ``logDt_range``, ``c_range`` is a
        ``(start, stop, step)`` triple consumed by ``np.arange``.

        Returns
        -------
        (dict_mat, param_mat) : cp.ndarray, cp.ndarray
            ``dict_mat`` has shape ``(n_chirplets, length)``; ``param_mat`` has
            shape ``(n_chirplets, 4)`` with columns ``[tc, fc, logDt, c]``.
        """
        if cache is None:
            cache = self.cache

        key = _params_to_key(tc_range, fc_range, logDt_range, c_range, self.length, self.FS)
        cache_addr = f"dict_cache_{key}.p"
        if self.debug:
            print(f"Cache addr: {cache_addr}", flush=True)

        if cache and os.path.exists(cache_addr):
            if self.debug:
                print("Loading dictionary from cache...", flush=True)
            dict_np, param_np = joblib.load(cache_addr)
            return cp.asarray(dict_np), cp.asarray(param_np)

        tc_vals    = np.arange(tc_range[0], tc_range[1], tc_range[2])
        fc_vals    = np.arange(fc_range[0], fc_range[1], fc_range[2])
        logDt_vals = np.arange(logDt_range[0], logDt_range[1], logDt_range[2])
        c_vals     = np.arange(c_range[0], c_range[1], c_range[2])

        dict_size = len(tc_vals) * len(fc_vals) * len(logDt_vals) * len(c_vals)
        if not self.mute:
            print(f"Generating dictionary of size {dict_size}...", flush=True)

        dict_mat_np  = np.zeros((dict_size, self.length), dtype=np.float32)
        param_mat_np = np.zeros((dict_size, 4),           dtype=np.float32)

        cnt = 0
        for tc in tc_vals:
            for fc in fc_vals:
                for logDt in logDt_vals:
                    for c in c_vals:
                        dict_mat_np[cnt]  = self.g_cpu(tc=tc, fc=fc, logDt=logDt, c=c)
                        param_mat_np[cnt] = np.array([tc, fc, logDt, c], dtype=np.float32)
                        if self.debug and cnt < 5:
                            print(f"Dict sample [{cnt}]: tc={tc}, fc={fc}, logDt={logDt}, c={c}", flush=True)
                        cnt += 1

        # single host->device transfer
        dict_mat  = cp.asarray(dict_mat_np)
        param_mat = cp.asarray(param_mat_np)

        if self.debug:
            print(f"Cache value: {cache}", flush=True)
        if cache:
            joblib.dump((dict_mat_np, param_mat_np), cache_addr)  # save np directly, no .get() needed
            if self.debug:
                print("Dictionary saved to cache.", flush=True)

        return dict_mat, param_mat

    # -------------------------------
    # Dictionary search
    # -------------------------------
    def search_dictionary_subset(
        self,
        signal: Union[np.ndarray, cp.ndarray],
        dict_mat: cp.ndarray,
        param_mat: cp.ndarray,
        top_k: int = 1,
    ) -> Tuple[List[int], List[float]]:
        """Project ``signal`` onto every (normalized) chirplet in ``dict_mat``
        (the family of chirplets) and return the indices/values of the
        ``top_k`` best matches."""
        if not isinstance(signal, cp.ndarray):
            sig = cp.asarray(signal)
        else:
            sig = signal

        dict_mat = dict_mat / cp.linalg.norm(dict_mat, axis=1, keepdims=True)
        projection_values = dict_mat.dot(sig)
        if top_k == 1:
            ind = int(cp.argmax(projection_values))
            val = float(cp.max(projection_values))
            if self.debug:
                print(f"Top-1 projection: idx={ind}, val={val}", flush=True)
            return [ind], [val]
        else:
            proj_np = projection_values.get()
            ind = np.argpartition(-proj_np, top_k - 1)[:top_k]
            indx = ind[np.argsort(-proj_np[ind])]
            val = proj_np[indx].tolist()
            if self.debug:
                print(f"Top-{top_k} projections: idxs={indx}, vals={val}", flush=True)
            return indx.tolist(), val

    # -------------------------------
    # Coarse-to-fine search
    # -------------------------------
    def coarser_to_finer(
        self,
        signal: Union[np.ndarray, cp.ndarray],
        coarse_range: Optional[List[List[float]]] = None,
        step_size: Optional[List[Tuple[float, float, float, float]]] = None,
        refine_levels: int = 2,
        top_k: int = 4,
        radius_steps: int = 2,
    ) -> np.ndarray:
        """Locate a good chirplet initialization via Latin-hypercube global
        search (level 0) followed by successive coarse-to-fine grid
        refinement around the current best candidates.

        Returns the best ``[tc, fc, logDt, c]`` parameter vector found.
        """
        self.progress = []

        sig = cp.asarray(signal) if not isinstance(signal, cp.ndarray) else signal

        if coarse_range is None:
            coarse_range = [[0, self.length], [0.7, 15.0], [-4.0, 4.0], [-30.0, 30.0]]

        if step_size is None:
            step_size = [
                (64, 2.0, 1.0, 6.0),
                (32, 1.0, 0.5, 3.0),
                (8, 0.5, 0.2, 1.0),
                (2, 0.1, 0.05, 0.25),
            ]

        candidates = None
        max_levels = min(len(step_size), refine_levels + 1)

        rad_tc    = radius_steps
        rad_fc    = radius_steps
        rad_logDt = max(1, radius_steps // 2)
        rad_c     = radius_steps

        for lev in range(0, max_levels):
            if not self.mute:
                print(f"--- Level {lev} ---", flush=True)

            if lev == 0:
                # LEVEL 0: global exploration via Latin hypercube sampling
                n_samples = 500
                bounds = [(r[0], r[1]) for r in coarse_range]
                lhs_params = self.lhs_sample(bounds, n_samples)

                # build on CPU, one transfer
                dict_mat_np = np.zeros((n_samples, self.length), dtype=np.float32)
                for i, (tc, fc, logDt, c) in enumerate(lhs_params):
                    dict_mat_np[i] = self.g_cpu(tc, fc, logDt, c)

                dict_mat  = cp.asarray(dict_mat_np)
                param_mat = cp.asarray(lhs_params, dtype=cp.float32)

                # compute projections and pick top_k
                dict_norms = cp.linalg.norm(dict_mat, axis=1)
                dict_norms = cp.where(dict_norms == 0, 1.0, dict_norms)
                proj = dict_mat.dot(sig) / dict_norms

                if top_k == 1:
                    ind  = int(cp.argmax(proj))
                    idxs = [ind]
                else:
                    proj_np         = proj.get()
                    idxs_unsorted   = np.argpartition(-proj_np, min(top_k, len(proj_np) - 1))[:top_k]
                    idxs_sorted     = idxs_unsorted[np.argsort(-proj_np[idxs_unsorted])]
                    idxs            = idxs_sorted.tolist()

                candidates = [param_mat[int(i)].get() for i in idxs]
                if not self.mute:
                    print(f"Level 0 coarse candidates (top {top_k}): {candidates}\n", flush=True)

                self.free_gpu(dict_mat)

            else:
                # REFINEMENT LEVELS: lev >= 1
                if not self.mute:
                    print(f"--- Refinement level {lev} ---", flush=True)
                next_candidates_all = []

                prev_steps = step_size[lev - 1]
                steps      = step_size[lev]

                for cand in candidates:
                    tc_c, fc_c, logDt_c, c_c = map(float, cand)

                    tc_r = (max(tc_c    - prev_steps[0] * rad_tc,    coarse_range[0][0]),
                            min(tc_c    + prev_steps[0] * rad_tc,    coarse_range[0][1]),
                            steps[0])

                    fc_r = (max(fc_c    - prev_steps[1] * rad_fc,    coarse_range[1][0]),
                            min(fc_c    + prev_steps[1] * rad_fc,    coarse_range[1][1]),
                            steps[1])

                    logDt_r = (max(logDt_c - prev_steps[2] * rad_logDt, coarse_range[2][0]),
                               min(logDt_c + prev_steps[2] * rad_logDt, coarse_range[2][1]),
                               steps[2])

                    c_r = (max(c_c - prev_steps[3] * rad_c, coarse_range[3][0]),
                           min(c_c + prev_steps[3] * rad_c, coarse_range[3][1]),
                           steps[3])
                    if self.debug:
                        print(f"Candidate center {cand}: tc_r={tc_r}, fc_r={fc_r}, "
                              f"logDt_r={logDt_r}, c_r={c_r}", flush=True)

                    dict_mat_f, param_mat_f = self.generate_dictionary_from_ranges(
                        tc_r, fc_r, logDt_r, c_r, cache=self.cache)

                    dict_norms_f = cp.linalg.norm(dict_mat_f, axis=1)
                    dict_norms_f = cp.where(dict_norms_f == 0, 1.0, dict_norms_f)
                    proj_f = dict_mat_f.dot(sig) / dict_norms_f

                    if top_k == 1:
                        best_idxs = [int(cp.argmax(proj_f))]
                    else:
                        proj_f_np   = proj_f.get()
                        best_idxs_u = np.argpartition(-proj_f_np, min(top_k, len(proj_f_np) - 1))[:top_k]
                        best_idxs   = best_idxs_u[np.argsort(-proj_f_np[best_idxs_u])].tolist()

                    for bi in best_idxs:
                        best_param = param_mat_f[int(bi)].get()
                        next_candidates_all.append(best_param)
                        chirplet       = self.normalized_chirplet(*best_param)
                        c_opt          = float(chirplet.dot(sig))
                        residual       = sig - c_opt * chirplet
                        residual_norm  = float(cp.linalg.norm(residual))
                        if self.debug:
                            print(f"  local best param: {best_param}, "
                                  f"residual norm (true cost): {residual_norm}", flush=True)

                    self.free_gpu(dict_mat_f)

                if self.debug:
                    print(f"Level {lev} coarse candidates (top {top_k}): {next_candidates_all}\n", flush=True)

                # rank all local-best candidates globally
                ranked = []
                for p in next_candidates_all:
                    a = self.normalized_chirplet(p[0], p[1], p[2], p[3])
                    ranked.append((p, float(a.dot(sig))))
                    del a

                if lev == 1:
                    max_keep = max(top_k, 16)
                elif lev == 2:
                    max_keep = max(top_k, 8)
                else:
                    max_keep = top_k

                ranked.sort(key=lambda x: -x[1])
                candidates = [r[0] for r in ranked[:max_keep]]
                if self.debug:
                    print(f"Refined global candidates after level {lev}: {candidates}\n", flush=True)
                self.progress.append(candidates[0].copy())

        best = candidates[0]
        if self.debug:
            print(f"Final best candidate: {best}\n", flush=True)
        return np.array(best, dtype=float)

    # -------------------------------
    # GPU memory helper
    # -------------------------------
    def free_gpu(self, *objects: Any) -> None:
        """Drop references to ``objects`` and release CuPy's memory pools."""
        for obj in objects:
            try:
                del obj
            except Exception:
                pass
        try:
            cp._default_memory_pool.free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    # -------------------------------
    # Full transform
    # -------------------------------
    def transform(
        self,
        signal: Union[np.ndarray, cp.ndarray],
        order: int = 5,
        debug: bool = False,
        coarse_range: Optional[List[List[float]]] = None,
        step_size: Optional[List[Tuple[float, float, float, float]]] = None,
        refine_levels: Optional[int] = None,
        radius_steps: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Decompose ``signal`` into ``order`` chirplets via matching pursuit.

        Each chirplet is initialized with ``coarser_to_finer`` and then locally
        polished with ``scipy.optimize.minimize``. Setting ``debug=True``
        enables verbose per-step diagnostics for this call (and any nested
        calls it makes) via ``self.debug``.

        Returns a dict with keys ``params``, ``coeffs``, ``approx``,
        ``residue``, and ``norm_residue`` (residual-to-signal norm ratio).
        """
        self.debug = debug

        sig        = cp.asarray(signal) if not isinstance(signal, cp.ndarray) else signal
        param_list = cp.zeros((order, 4),       dtype=cp.float32)
        coeff_list = cp.zeros(order,             dtype=cp.float32)
        approx     = cp.zeros(len(signal),       dtype=cp.float32)
        residue    = cp.copy(sig)
        chirp_list = [cp.zeros_like(sig) for _ in range(order)]

        for P in range(order):
            if not self.mute:
                print(f"\n--- Extracting chirplet {P + 1} ---", flush=True)

            # Step 1: coarse-to-fine initialization
            init_params = self.coarser_to_finer(
                residue,
                coarse_range=coarse_range,
                step_size=step_size,
                refine_levels=refine_levels,
                radius_steps=radius_steps,
                top_k=top_k
            )

            if self.debug:
                print(f"Initial params for optimization: {init_params}", flush=True)

            # Step 2: parameter optimization
            res        = optimize.minimize(self.minimize_this, init_params, args=(residue.get(),))
            new_params = cp.array(res.x)

            # Step 3: generate the chirplet
            updated_base_chirplet = self.g(
                tc=new_params[0], fc=new_params[1],
                logDt=new_params[2], c=new_params[3]
            )

            updated_base_chirplet /= cp.linalg.norm(updated_base_chirplet)

            updated_chirplet_coeff = cp.vdot(updated_base_chirplet, residue)

            # Step 6: update lists
            chirp_list[P] = updated_base_chirplet
            coeff_list[P] = updated_chirplet_coeff
            param_list[P] = new_params

            # Step 7: update residue and approximation
            approx  = cp.sum(cp.stack([coeff_list[i] * chirp_list[i] for i in range(P + 1)]), axis=0)
            residue = sig - approx

            residue_norm = cp.linalg.norm(residue).item()
            signal_norm  = cp.linalg.norm(sig).item()

        return {
            "params":  param_list.get(),
            "coeffs":  coeff_list.get(),
            "approx":  approx.get(),
            "residue": residue.get(),
            "norm_residue": residue_norm / signal_norm
        }

    def minimize_this(self, coeffs: Sequence[float], signal: np.ndarray) -> float:
        """Objective for ``scipy.optimize.minimize``: negative absolute
        correlation between a candidate chirplet and ``signal`` (CPU-side)."""
        chirplet = self.g_cpu(tc=coeffs[0], fc=coeffs[1], logDt=coeffs[2], c=coeffs[3])
        return -1 * abs(np.dot(chirplet, signal))

    # -------------------------------
    # Visualization
    # -------------------------------
    def visualize_candidate(
        self,
        tc: float, fc: float, logDt: float, c: float,
        level: int = 0, iteration: int = 0,
        signal: Optional[Union[np.ndarray, cp.ndarray]] = None,
        kind: str = "candidate",
        idx: Optional[int] = None,
        coeff: Optional[complex] = None,
    ) -> None:
        """Live-plot a chirplet's time/frequency footprint as an ellipse.

        ``kind`` controls styling: ``"background"`` draws a faint reference
        marker, ``"candidate"`` draws a fading trail of search candidates,
        and ``"extracted"`` draws a permanent, labeled marker for chirplets
        that have been accepted into the decomposition.
        """

        def to_num(x):
            if isinstance(x, cp.ndarray):
                return float(x.get()) if x.size == 1 else cp.asnumpy(x)
            try:
                return float(x)
            except Exception:
                return x

        tc_n, fc_n     = to_num(tc),    to_num(fc)
        logDt_n, c_n   = to_num(logDt), to_num(c)

        time_scale  = max(1.0, self.length / 50.0)
        fc_max      = getattr(self, "fc_display_max", 30.0)
        freq_scale  = max(1.0, fc_max / 10.0)
        width       = np.exp(logDt_n) * time_scale
        height      = (abs(c_n) + 1e-8) * freq_scale

        cmap = plt.cm.plasma
        if kind == "background":
            edgecolor, alpha, lw, z = "black", 0.25, 1, 0
        elif kind == "extracted":
            edgecolor, alpha, lw, z = plt.cm.tab10((idx or 0) % 10), 0.9, 2.5, 3
        else:
            edgecolor, alpha, lw, z = cmap((level % 6) / 6.0), 0.8, 1.6, 2

        if not hasattr(self, "_vis_initialized"):
            plt.ion()
            self._fig_vis, self._ax_vis = plt.subplots(figsize=(10, 5))
            self._ellipse_patches   = []
            self._candidate_patches = []
            self._extracted_patches = []
            ax = self._ax_vis
            ax.set_xlabel("Time center (tc)")
            ax.set_ylabel("Frequency center (fc)")
            ax.set_title("Chirplet Candidate Ellipses")
            ax.grid(True)
            ax.set_xlim(0, self.length)
            ax.set_ylim(0, fc_max)
            self._vis_initialized = True
        else:
            ax     = self._ax_vis
            fc_max = self._ax_vis.get_ylim()[1]

        if kind == "background":
            bg_e = Ellipse((tc_n, fc_n), width=width, height=height,
                           edgecolor=edgecolor, facecolor='none', lw=lw, alpha=alpha, zorder=z)
            bg_e._is_background = True
            ax.add_patch(bg_e)
            self._ellipse_patches.append(bg_e)
            ax.scatter(tc_n, fc_n, color=edgecolor, s=30, marker='o', zorder=z + 1)
            self._bg_added = True
            return

        e = Ellipse((tc_n, fc_n), width=width, height=height,
                    edgecolor=edgecolor, facecolor='none', lw=lw, alpha=alpha, zorder=z)
        if kind == "candidate":
            e._kind = "candidate"
            self._candidate_patches.append(e)
        else:
            e._kind  = "extracted"
            e._idx   = idx
            e._coeff = coeff
            self._extracted_patches.append(e)
            if idx is not None:
                txt = f"{idx}"
                if coeff is not None:
                    try:
                        mag  = abs(complex(coeff))
                        txt += f":{mag:.2f}"
                    except Exception:
                        pass
                ax.text(tc_n, fc_n, txt, fontsize=9, zorder=z + 1, ha='center', va='bottom')

        ax.add_patch(e)
        self._ellipse_patches.append(e)
        ax.scatter(tc_n, fc_n, color=edgecolor, s=30, marker='o', zorder=z + 1)

        decay     = 0.8
        min_alpha = 0.05
        new_candidate_list = []
        for p in list(self._candidate_patches):
            if getattr(p, "_is_background", False) or getattr(p, "_kind", None) != "candidate":
                continue
            curr_alpha = p.get_alpha()
            new_alpha  = max(min_alpha, curr_alpha * decay)
            p.set_alpha(new_alpha)
            if new_alpha > min_alpha + 1e-6:
                new_candidate_list.append(p)
            else:
                try:
                    p.remove()
                    self._ellipse_patches.remove(p)
                except Exception:
                    pass
        self._candidate_patches = new_candidate_list

        ax.set_xlim(0, self.length)
        ax.set_ylim(-fc_max, fc_max)
        plt.pause(0.5)

    # -------------------------------
    # Helpers
    # -------------------------------
    def lhs_sample(self, param_ranges: Sequence[Tuple[float, float]], n_samples: int) -> np.ndarray:
        """Draw ``n_samples`` points via Latin hypercube sampling, scaled to
        the ``(low, high)`` bounds in ``param_ranges``."""
        sampler = qmc.LatinHypercube(d=len(param_ranges))
        sample  = sampler.random(n=n_samples)
        return qmc.scale(sample,
                         [r[0] for r in param_ranges],
                         [r[1] for r in param_ranges])

    def normalized_chirplet(self, tc: float, fc: float, logDt: float, c: float) -> cp.ndarray:
        """Generate a unit-norm chirplet (falls back to the raw chirplet if
        its norm is zero, to avoid a division-by-zero)."""
        chirplet = self.g(tc=tc, fc=fc, logDt=logDt, c=c)
        norm = float(cp.linalg.norm(chirplet).get())
        if norm == 0:
            return chirplet
        return (chirplet / norm).astype(cp.float32)