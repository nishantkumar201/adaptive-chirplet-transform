# Adaptive Chirplet Transform (ACT) – CPU & GPU Reference Implementations

## Overview

This repository provides CPU and GPU reference implementations of the Adaptive Chirplet Transform (ACT).

It includes:
- A CPU reference implementation (act.py) for correctness verification and reproducibility
- A GPU-accelerated implementation (act_gpu.py) leveraging **NVIDIA CUDA (via [CuPy](https://cupy.dev))** to accelerate dictionary generation and iterative decomposition

Both implementations include mathematically corrected normalization and expanded evaluation support, ensuring numerical consistency across backends.

The codebase is intended for:
- Reproducible ACT research
- Performance comparison between CPU and GPU pipelines

The GPU implementation has been previously validated and published in a conference setting. The CPU implementation has been updated to match the corrected mathematical formulation used in the GPU version. Optional CPU/GPU monitoring utilities are included to record power, memory, and utilization metrics.

This implementation is based on the original CPU code by [amanb2000](https://github.com/amanb2000/Adaptive_Chirplet_Transform).

---
## Repository Structure
The repository currently includes four core modules:

- **`act.py`**- Implements the mathematically correct Adaptive Chirplet Transform
- **`act_gpu.py`** – Implements the Adaptive Chirplet Transform including GPU-accelerated dictionary generation and signal decomposition. 
- **`monitoringclass.py`** – Provides CPU/GPU usage monitoring (power, memory, utilization), logging results to CSV for later inspection.  
- **`run_act_example.py`** – Example pipeline for loading EEG data, preprocessing, applying ACT, and exporting results.  

---
## Features

- CPU reference implementation (act.py) for correctness verification and reproducibility
- GPU-accelerated chirplet dictionary construction using CuPy (act_gpu.py)
- Iterative ACT decomposition with parameter refinement via SciPy optimization  
- EEG preprocessing supported through [MNE](https://mne.tools/)  
- Optional CPU/GPU monitoring during runs  
- Outputs results (parameters, coefficients, reconstruction errors) as CSV  

---

## Requirements

- **Python** 3.9+  
- **NVIDIA GPU** with CUDA support (tested with CUDA 12.x) (if doing act_gpu.py)
- Recommended: 8 GB+ VRAM for larger EEG datasets (if doing act_gpu.py)

### Python Dependencies
All required packages are listed in `requirements.txt`.  
Install them with:
~~~bash
pip install -r requirements.txt
~~~

## Quick Start

This guide will help you set up and run the provided code examples quickly.

---

### 1. Clone the Repository

~~~bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
~~~

### 2.	Prepare EEG data
Place .edf EEG files in the expected directory structure.
Example path used in run_act_example.py:

~~~
ACT/Bitbrain/sub-1/eeg/sub-1_task-Sleep_acq-headband_eeg.edf
~~~

### 3. Run the example script

~~~
python run_act_example.py
~~~

### 	4.	Check outputs
Result Logs:
~~~
act_results_sub-1_optimized.csv
~~~
Monitoring logs:
~~~
cpu_monitoring.csv
gpu_monitoring.csv
~~~

## Output Format

The output CSV contains:

| Epoch | Params (tc, fc, logDt, c) | Coeffs | Error | Residue |
|-------|----------------------------|--------|-------|---------|

## Repository Structure
~~~
├── act.py                 # CPU reference implementation
├── act_gpu.py             # GPU-accelerated ACT (CuPy / CUDA)
├── monitoringclass.py     # CPU/GPU monitoring utilities
├── run_act_example.py     # Example EEG processing pipeline
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── LICENSE                # MIT License
~~~

## Citation

Preliminary results of this work appeared in the  
*27th Annual Mersivity / Water-HCI Symposium Proceedings*  
(pp. 55–56), Zenodo, 2025.  
👉 [https://doi.org/10.5281/zenodo.16973160](https://doi.org/10.5281/zenodo.16973160)

A full version with extended profiling and evaluation has been published at  
the *International Conference on Sensing Technology (ICST) 2025*.
👉 [https://doi.org/10.1109/ICST66402.2025.11512436](https://doi.org/10.1109/ICST66402.2025.11512436)

If you use this repository (CPU or GPU code), please additionally cite: 
Nishant Kumar, Adaptive Chirplet Transform (ACT) – CPU/GPU Reference Implementation, GitHub repository, 2026.

## Author
Nishant Kumar

### Zenodo Preliminary Results
~~~ bibtex
@inproceedings{Mersivity2025,
  editor    = {Steve Mann and Michael Condry and Nishant Kumar},
  title     = {27th Annual Mersivity / Water-HCI Symposium Proceedings},
  pages     = {55--56},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.16973160},
  url       = {https://doi.org/10.5281/zenodo.16973160}
}
~~~

## License
This project is released under the MIT License.


