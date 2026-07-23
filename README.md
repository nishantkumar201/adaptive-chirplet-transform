# Adaptive Chirplet Transform (ACT) – CPU & GPU Reference Implementations

## Overview

This repository provides CPU and GPU reference implementations of the Adaptive Chirplet Transform (ACT).

It includes:

- A **CPU reference implementation** (`act.py`) for correctness verification and reproducibility
- A **GPU-accelerated implementation using CUDA / CuPy** (`act_gpu_cuda.py`), supporting a **hybrid mode** in which chirplet dictionaries are generated on CPU (NumPy) and uploaded to GPU, or generated directly on GPU
- A **GPU-accelerated implementation using PyTorch** (`act_gpu_pytorch.py`), mirroring the CUDA/CuPy version's hybrid dictionary-generation strategy but built on PyTorch tensors
- A **Latin-hypercube / coarse-to-fine matching pursuit variant** (`act_lem.py`), which locates chirplet initializations via global Latin hypercube sampling followed by successive coarse-to-fine grid refinement, then polishes them with local gradient-based optimization

All implementations share a common mathematical formulation and normalization convention, ensuring numerical consistency across backends.

The codebase is intended for:

- Reproducible ACT research
- Performance comparison between CPU and GPU pipelines (CUDA vs. PyTorch, hybrid vs. non-hybrid dictionary generation)

This implementation is based on the original CPU code by [amanb2000](https://github.com/amanb2000/Adaptive_Chirplet_Transform).

---

## Old vs. New: Reconstruction Quality

![ACT Old vs. ACT New reconstruction and residual comparison on EEG data](Images/ACT_EEG.pdf)

The top panel overlays the original EEG waveform against reconstructions from the original (`ACT Old`) and corrected (`ACT New`) implementations; the bottom panel shows the corresponding residuals. The old implementation systematically over- and under-shoots the signal's peaks and troughs and flattens out over the later plateau, leaving a residual that swings persistently in both directions. The corrected implementation tracks the original waveform far more tightly throughout, including through the amplitude changes near the end, and its residual stays close to zero rather than oscillating with the signal itself, indicating the fix addresses a systematic bias in the reconstruction, not just noise. Full quantitative comparisons (multiple signals, error metrics, runtime) are in the arXiv preprint and the upcoming OJSP paper (see Citation below).

---

## Repository Structure

```
├── Testing Scripts
│   ├── Data
│   │   ├── data_SAAB_SIRS_77GHz_FMCW.npy               # data file not included, see below in Data Section
│   │   ├── sub-1_task-Sleep_acq-headband_eeg.edf       # data file not included, see below in Data Section
│   │   ├── S2_E3_A1_basic_movement.mat                 # data file not included, see below in Data Section
│   │   └── chb01_01.edf                                # data file not included, see below in Data Section
│   ├── OJSP Testing
│   │   ├── Original_Code
│   │   │   └── old_act.py                              # Code from amanb2000
│   │   ├── act_comparison_all.ipynb
│   │   ├── act_lem_comparison.ipynb
│   │   ├── act_multi_channel.ipynb
│   │   ├── comparisonplot_hybrid_CPU.ipynb
│   │   ├── lem_comparison_plot.ipynb
│   │   ├── ordervsresidueplot.ipynb
│   │   └── residual.ipynb
│   └── run_act.ipynb
├── act.py                 # CPU reference implementation
├── act_gpu_cuda.py         # GPU-accelerated ACT via CUDA/CuPy (hybrid + full-GPU dictionary generation)
├── act_gpu_pytorch.py      # GPU-accelerated ACT via PyTorch (hybrid + full-GPU dictionary generation)
├── act_lem.py              # Latin-hypercube / coarse-to-fine matching pursuit ACT variant
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── LICENSE                 # MIT License
```

---

## Features

- CPU reference implementation (`act.py`) for correctness verification and reproducibility
- GPU-accelerated chirplet dictionary construction via CuPy/CUDA (`act_gpu_cuda.py`) or PyTorch (`act_gpu_pytorch.py`)
- **Hybrid dictionary generation**: chirplets can be built on CPU (NumPy) then transferred to GPU in a single batch, or generated directly on GPU — selectable per run
- Iterative ACT decomposition (matching pursuit) with local parameter refinement via SciPy optimization
- Latin-hypercube global search with coarse-to-fine grid refinement (`act_lem.py`) for improved initialization robustness
- Multi-channel / batched signal support
- Dictionary caching to disk (via `joblib`) to avoid regenerating identical parameter grids
- Outputs results (parameters, coefficients, reconstruction errors, residues) as CSV / in-memory arrays

---

## Requirements

- **Python** 3.9+
- **NVIDIA GPU** with CUDA support (tested with CUDA 12.x), required for `act_gpu_cuda.py` and `act_lem.py`
- **PyTorch with CUDA support**, required for `act_gpu_pytorch.py`
- Recommended: 8 GB+ VRAM for larger EEG datasets

### Python Dependencies

Core (CPU-only) dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

GPU-specific packages are commented out in `requirements.txt` since they depend on your CUDA setup, so uncomment and install as needed:

```bash
pip install cupy-cuda12x   # for act_gpu_cuda.py / act_lem.py — adjust '12x' to match your CUDA version
pip install torch          # for act_gpu_pytorch.py — install the CUDA-enabled build for your platform
```

## Quick Start

Two steps to get a working example running end-to-end:

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/nishantkumar201/adaptive-chirplet-transform.git
cd adaptive-chirplet-transform
pip install -r requirements.txt
```

### 2. Run the example notebook

```
Testing Scripts/run_act.ipynb
```

This notebook runs the full pipeline end-to-end on the included Bitbrain sleep EEG data: dictionary generation, decomposition, and plotting. This uses the included `sub-1_task-Sleep_acq-headband_eeg.edf` file (no data download needed to try it). It's meant as a **template** to copy from, not a fixed pipeline: it's the fastest way to see how `order`, dictionary parameter ranges, and backend choice (hybrid vs. non-hybrid, CUDA vs. PyTorch) affect the decomposition and residual error.

For results across all four datasets and side-by-side backend comparisons (CUDA vs. PyTorch, hybrid vs. non-hybrid, LEM vs. grid search), see the notebooks in `Testing Scripts/OJSP Testing/`, e.g. `act_comparison_all.ipynb`, `act_lem_comparison.ipynb`, `act_multi_channel.ipynb`, `ordervsresidueplot.ipynb`.

---

## Using ACT on Your Own Signal

Every backend follows the same basic pattern: instantiate the class with your signal's sampling rate/length and your chosen chirplet parameter ranges, then call `transform()`:

```python
from act import ACT  # or: from act_gpu_cuda import ACT / from act_gpu_pytorch import ACT

model = ACT(
    FS=256,                    # sampling rate of YOUR signal
    length=3840,               # length of YOUR signal, in samples
    tc_info=(0, 3840, 1),      # time-center search range (start, stop, step)
    fc_info=(0.7, 15, 0.2),    # frequency-center search range, in Hz
    logDt_info=(-4, -1, 0.3),  # log-duration search range
    c_info=(-30, 30, 3),       # chirp-rate search range
)

result = model.transform(my_signal, order=5)  # decompose into 5 chirplets
```

`act_lem.py` (`ACT_LEM`) follows the same general idea, but the parameter search works differently: `tc_info`, `fc_info`, `logDt_info`, and `c_info` passed to the constructor are reserved metadata only (kept for API compatibility with the other backends), the actual search ranges are passed to `transform()` itself via `coarse_range` and `step_size`, which drive the Latin-hypercube global search and coarse-to-fine grid refinement:

```python
from act_lem import ACT_LEM

FS = 256
EPOCH_LENGTH = 2 * FS  # 512 samples

model = ACT_LEM(
    FS=FS, length=EPOCH_LENGTH,
    tc_info=(0, EPOCH_LENGTH, 16),   # reserved metadata, not used for the search itself
    fc_info=(0.5, 15, 0.5),
    logDt_info=(-4, 1, 0.5),
    c_info=(-10, 10, 0.5),
    mute=True,
)

result = model.transform(
    my_epoch,
    order=10,
    refine_levels=2,
    top_k=4,
    radius_steps=2,
    coarse_range=[[0, EPOCH_LENGTH], [0.5, 15.0], [-4.0, 1.0], [-10.0, 10.0]],
    step_size=[
        (64, 2.0, 2.0, 5.0),
        (32, 1.0, 1.0, 2.5),
        (16, 0.5, 0.5, 0.5),
    ],
)
```

This mirrors the CPU-vs-GPU-vs-LEM benchmark methodology used for the OJSP submission (see `Testing Scripts/OJSP Testing/`): the same signal/epoch is run through `act.py`, `act_gpu_cuda.py` (with `hybrid=True`, i.e. dictionary built on CPU and uploaded to GPU), and `act_lem.py` under matched `FS`/`length`/`order` settings, and runtime + reconstruction quality (`norm_residue`) are compared across repeats. This is a good template if you want to benchmark backends against each other on your own data rather than just picking one.

This mirrors the setup used in the OJSP benchmarking notebooks (`Testing Scripts/OJSP Testing/`), which run the CPU, GPU-hybrid, and LEM backends side by side over repeated epochs to compare runtime and residual error. See `act_comparison_all.ipynb` and `act_lem_comparison.ipynb` for the full benchmark harness.

**A note on parameter ranges:** the `fc_info`, `logDt_info`, and `c_info` values above are tuned for EEG-scale sleep data and will **not** transfer directly to radar, EMG, or other signal types. Choosing good ranges for a new signal is inherently a bit of trial and error. Start with ranges spanning your signal's expected frequency content and time scale, run `transform()` with a small `order` (e.g. 3–5), then check `norm_residue` and plot the reconstruction to confirm the dictionary is actually capturing the signal's structure before scaling up. `ordervsresidueplot.ipynb` and `residual.ipynb` under `Testing Scripts/OJSP Testing/` walk through exactly this diagnostic process and are a good starting point for tuning against a new signal type.

## Data

The `Testing Scripts/Data/` folder references four datasets used for validation across different signal domains (EEG, radar, EMG). Only the **Bitbrain sleep EEG dataset** is kept/included directly in this repository; the others must be downloaded separately from their sources below.

| File                                    | Dataset                                              | Included in repo?                                              |
| --------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------- |
| `sub-1_task-Sleep_acq-headband_eeg.edf` | Bitbrain Open Access Sleep Dataset                   | ✅ Yes                                                         |
| `chb01_01.edf`                          | CHB-MIT Scalp EEG Database                           | ❌ No, download from PhysioNet                                 |
| `data_SAAB_SIRS_77GHz_FMCW.npy`         | Raw Radar FMCW Dataset                               | ❌ No, download from IEEE DataPort                             |
| `S2_E3_A1_basic_movement.mat`           | Non-invasive EMG dataset for robotic hand prostheses | ❌ No, download from the associated Scientific Data repository |

### EEG (sleep) — Bitbrain Open Access Sleep Dataset

```bibtex
@misc{ds005555:1.1.0,
  author       = {Eduardo López-Larraz and María Sierra-Torralba and Sergio Clemente and Galit Fierro and David Oriol and Javier Mínguez and Luis Montesano and Jens G. Klinzing},
  title        = {{Bitbrain Open Access Sleep Dataset}},
  year         = {2025},
  publisher    = {OpenNeuro},
  doi          = {10.18112/openneuro.ds005555.v1.1.0},
  url          = {https://doi.org/10.18112/openneuro.ds005555.v1.1.0}
}
```

### EEG (seizure) — CHB-MIT Scalp EEG Database

```bibtex
@article{PhysioNet-chbmit-1.0.0,
  author = {Guttag, John},
  title = {{CHB-MIT Scalp EEG Database}},
  journal = {{PhysioNet}},
  year = {2010},
  month = jun,
  note = {Version 1.0.0},
  doi = {10.13026/C2K01R},
  url = {https://doi.org/10.13026/C2K01R}
}
```

### Radar — Raw Radar FMCW Dataset

```bibtex
@misc{41e8-8v73-25,
  author       = {Wissal Zarrami and Guillaume-Alexandre Bilodeau},
  title        = {{Raw Radar FMCW Dataset}},
  year         = {2025},
  publisher    = {IEEE DataPort},
  doi          = {10.21227/41e8-8v73},
  url          = {https://dx.doi.org/10.21227/41e8-8v73}
}
```

### EMG — Non-invasive naturally-controlled robotic hand prostheses dataset

```bibtex
@article{Atzori_Gijsberts_Castellini_Caputo_Hager_Elsig_Giatsidis_Bassetto_Müller_2014,
  title={Electromyography data for non-invasive naturally-controlled robotic hand prostheses},
  volume={1},
  DOI={10.1038/sdata.2014.53},
  number={1},
  journal={Scientific Data},
  author={Atzori, Manfredo and Gijsberts, Arjan and Castellini, Claudio and Caputo, Barbara and Hager, Anne-Gabrielle Mittaz and Elsig, Simone and Giatsidis, Giorgio and Bassetto, Franco and Müller, Henning},
  year={2014},
  month={Dec}
}
```

---

## Output Format

Decomposition output (from `transform()` in any backend) contains, per signal:

| Params (tc, fc, logDt, c) | Coeffs | Approx | Residue | Norm Residue / Error |
| ------------------------- | ------ | ------ | ------- | -------------------- |

---

## Citation

This work is currently in progress toward submission to **OJSP** (Open Journal of Signal Processing), expected within the next 2–3 days. The OJSP citation will be added here once the article is published.

A preprint is available on arXiv:

```bibtex
@misc{kumar2026stabledeployableadaptivechirplet,
      title={Toward a Stable and Deployable Adaptive Chirplet Transform: Residual Projection, Hybrid GPU Acceleration, and Multi-Channel Scalability},
      author={Nishant Kumar and Steve Mann},
      year={2026},
      eprint={2607.16629},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2607.16629},
}
```

Preliminary results of this work appeared in the
_27th Annual Mersivity / Water-HCI Symposium Proceedings_
(pp. 55–56), Zenodo, 2025.
👉 [https://doi.org/10.5281/zenodo.16973160](https://doi.org/10.5281/zenodo.16973160)

A full version with extended profiling and evaluation has been published at
the _International Conference on Sensing Technology (ICST) 2025_.
👉 [https://ieeexplore.ieee.org/document/11512436](https://ieeexplore.ieee.org/document/11512436)

If you use this repository (CPU or GPU code), please additionally cite:
Nishant Kumar, Adaptive Chirplet Transform (ACT) – CPU/GPU Reference Implementation, GitHub repository, 2026.

## Author

Nishant Kumar

### Zenodo Preliminary Results

```bibtex
@inproceedings{Mersivity2025,
  editor    = {Steve Mann and Michael Condry and Nishant Kumar},
  title     = {27th Annual Mersivity / Water-HCI Symposium Proceedings},
  pages     = {55--56},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.16973160},
  url       = {https://doi.org/10.5281/zenodo.16973160}
}
```

### ICST Results

```bibtex
@INPROCEEDINGS{11512436,
  author={Kumar, Nishant and Mann, Steve},
  booktitle={2025 18th International Conference on Sensing Technology (ICST)},
  title={GPU-Accelerated Chirplet Transform: Scalable Runtime Profiling and Analysis},
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Graphics processing units;Timing;Central Processing Unit;Memory;Dictionaries;Testing;Transforms;Electroencephalography;Modeling;Printing;Chirplet Transform;Hardware Acceleration;GPGPU;Unified Memory;Time-Frequency Analysis;EEG Signal Processing;Signal Decomposition},
  doi={10.1109/ICST66402.2025.11512436}}
```

## License

This project is released under the MIT License.
