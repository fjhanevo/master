import gc
from time import time
import numpy as np
from vector_match import vector_match

if __name__ == '__main__':
    DIR_NPY = 'npy_files/'
    FILE = 'LF_r_theta_sim.npy'
    FILE_INTENSITY = 'sim_r_theta_intensity.npy'
    step = 0.5 # degrees
    reciprocal_radius = 1.35 # Å^-1

    # load data 
    sim_data = np.load(DIR_NPY+FILE, allow_pickle=True)
    sim_data_intensity = np.load(DIR_NPY+FILE_INTENSITY, allow_pickle=True)

    n_best = len(sim_data)

    # filter the data to make it inhomogeneous
    sim_filtered= []
    for idx in range(sim_data.shape[0]):
        r_theta = sim_data[idx]
        # mask out zeroes
        mask = ~np.all(r_theta == 0, axis=1)
        # apply mask
        r_theta_masked = r_theta[mask]
        sim_filtered.append(r_theta_masked)
    sim_filtered = np.array(sim_filtered, dtype=object)

    sim_filtered_intensity = []
    for idx in range(sim_data_intensity.shape[0]):
        r_theta_intensity = sim_data_intensity[idx]
        # mask out zeroes
        mask = ~np.all(r_theta_intensity == 0, axis=1)
        r_theta_intensity_masked = r_theta_intensity[mask]
        sim_filtered_intensity.append(r_theta_intensity_masked)
    sim_filtered_intensity = np.array(sim_filtered_intensity, dtype=object)

    ### File to test sum_score
    t1 = time()
    filename="280525_VM_sum_score_algorithm_accuracy.npy"
    result = vector_match(
        experimental=sim_filtered,
        simulated=sim_data,
        step_size=step,
        reciprocal_radius=reciprocal_radius,
        n_best=n_best,
        method="sum_score"
    )
    np.save(file=filename, arr=result, allow_pickle=True)
    # free memory
    del result
    gc.collect()
    t2 = time()
    print(f"Computation time for {filename}: {(t2-t1)/3600} hrs")

    
    ### File for sum_score_weighted
    t1 = time()
    filename="280525_VM_sum_score_weighted_algorithm_accuracy.npy"
    result = vector_match(
        experimental=sim_filtered,
        simulated=sim_data,
        step_size=step,
        reciprocal_radius=reciprocal_radius,
        n_best=n_best,
        method="sum_score_weighted",
        distance_bound=0.05
    )
    np.save(file=filename, arr=result, allow_pickle=True)
    # free memory
    del result
    gc.collect()
    t2 = time()
    print(f"Computation time for {filename}: {(t2-t1)/3600} hrs")

    ### File for ang_score
    t1 = time()
    filename="280525_VM_ang_score_algorithm_accuracy.npy"
    result = vector_match(
        experimental=sim_filtered,
        simulated=sim_data,
        step_size=step,
        reciprocal_radius=reciprocal_radius,
        n_best=n_best,
        method="score_ang",
        ang_thresh_rad=0.05 
    )
    np.save(file=filename, arr=result, allow_pickle=True)
    # free memory
    del result
    gc.collect()
    t2 = time()
    print(f"Computation time for {filename}: {(t2-t1)/3600} hrs")

    ### File for score_intensity
    t1 = time()
    filename="280525_VM_score_intensity_algorithm_accuracy.npy"
    result = vector_match(
        experimental=sim_filtered_intensity,
        simulated=sim_data_intensity,
        step_size=step,
        reciprocal_radius=reciprocal_radius,
        n_best=n_best,
        method="score_intensity",
        distance_bound=0.05

    )
    # free memory
    np.save(file=filename, arr=result, allow_pickle=True)
    del result
    gc.collect()
    t2 = time()
    print(f"Computation time for {filename}: {(t2-t1)/3600} hrs")

    ### File for fast sum_score
    t1 = time()
    filename="280525_VM_FAST_sum_score_algorithm_accuracy.npy"
    result = vector_match(
        experimental=sim_filtered,
        simulated=sim_data,
        step_size=step,
        reciprocal_radius=reciprocal_radius,
        n_best=n_best,
        method="sum_score",
        fast=True
    )
    # free memory
    np.save(file=filename, arr=result, allow_pickle=True)
    del result
    gc.collect()
    t2 = time()
    print(f"Computation time for {filename}: {(t2-t1)/3600} hrs")

