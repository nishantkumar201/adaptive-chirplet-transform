import act_gpu as act_lib
import numpy as np
import cupy as cp
import pandas as pd
import mne
import csv
import os
import time
from cupy.cuda import stream, memory
import monitoringclass

start_time = time.time()

# Enable Unified Memory to allow dynamic memory management                                 
cp.cuda.set_allocator(memory.malloc_managed)


# Initialize monitoring class
monitoring = monitoringclass.MonitoringClass()

epoch = 5
act = act_lib.ACT(
    FS=256,
    length=epoch * 256,
    tc_info=(0, epoch * 256, 64),
    fc_info=(0.6, 15, 1),
    logDt_info=(-4, 0, 0.3),
    c_info=(-10, 10, 0.75),
    force_regenerate=True,
    mute=False,
    monitor = True
)

# Load EEG data
data_file = os.path.join(os.getcwd(), "ACT/Bitbrain/sub-1/eeg/sub-1_task-Sleep_acq-headband_eeg.edf")
raw_data = mne.io.read_raw_edf(data_file, preload=True, verbose=False)
raw_data.pick_channels(["HB_1", "HB_2"])
raw_data.notch_filter(freqs=50, fir_design="firwin", verbose=False)

eeg_data = raw_data.get_data().T  # Shape (samples, channels)
eeg_data_gpu = cp.asarray(eeg_data, dtype=cp.float32) 


epoch_length = epoch * act.FS

# num_epochs = eeg_data.shape[0] // epoch_length # When doing the entire dataset
num_epochs = 1
output_csv = "act_results_sub-1_optimized.csv"

with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Params", "Coeffs", "Error", "Residue"])
    
    for epoch_idx in range(num_epochs):
        
        if monitor:
            monitoring.start_CPU_monitoring()
            monitoring.start_GPU_monitoring()

        start_idx = epoch_idx * epoch_length
        end_idx = start_idx + epoch_length
        print(f"Processing epoch {epoch_idx + 1}/{num_epochs}")

        for electrode_idx, electrode_name in enumerate(["HB_1", "HB_2"]):
            segment_gpu = eeg_data_gpu[start_idx:end_idx, electrode_idx]

            # Use a stream for parallelism, but make sure to synchronize
            my_stream = cp.cuda.Stream()
            with my_stream:
                result = act.transform(segment_gpu, order=6, debug=False)
            my_stream.synchronize()  # Make sure the GPU has completed work

            # Move results to CPU
            params = cp.asnumpy(result["params"]).tolist()
            coeffs = cp.asnumpy(result["coeffs"]).tolist()
            residue = cp.asnumpy(result["residue"]).tolist()

            writer.writerow([epoch_idx+1, params, coeffs, result["error"], residue])
        if monitor:
            monitoring.stop_CPU_monitoring()
            monitoring.stop_GPU_monitoring()

                
    print(f"Epoch {epoch_idx} completed in {time.time() - start_time:.2f} sec")

print(f"Total processing time: {time.time() - start_time:.2f} sec")
