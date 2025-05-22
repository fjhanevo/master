import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from numba import njit
import heapq
from joblib import Parallel, delayed
from vm_utils import *

"""
Fil for sphere matching!:)
"""

def sum_score(
    exp3d: np.ndarray, 
    exp3d_mirror: np.ndarray, 
    sim_trees,
    step_size_rad: float
) -> tuple:
    """
    Docstring, dette er score A
    """
    best_score, best_rotation, mirror = np.inf, 0.0, 0

    for rot_idx, sim_tree in enumerate(sim_trees):
        # get total points for normalisation
        sim_points = sim_tree.data
        n_total = len(exp3d) + len(sim_points)

        # skip if 0 vectors are found for HMS safety guidelines
        if n_total == 0:
            continue

        # calculate nn-distances
        distances, _ = sim_tree.query(exp3d)
        distances_mirror, _ = sim_tree.query(exp3d_mirror)

        # calculate scores
        scores = [
            (np.sum(distances) / n_total, 1),
            (np.sum(distances_mirror) / n_total, -1),
        ]

        # check score and keep only the best score for each sim_frame
        for score, mirror_flag in scores:
            if score < best_score:
                best_score = score
                ang = rot_idx * step_size_rad
                mirror = mirror_flag
                best_rotation = wrap_degrees(ang, mirror)

    return best_score, best_rotation, mirror

def vector_match(
    experimental: np.ndarray,
    simulated: np.ndarray,
    step_size: float, 
    reciprocal_radius: float,
    n_best: int,
    method: str = "sum",
    **kwargs
) -> np.ndarray:
    """
    Docstring
    """
    
    # Convert input degrees to radians
    step_size_rad = np.deg2rad(step_size)
    if method == "sum" or method == "kd":
        # Precompute KD-trees for rotated simulated frames
        precomputed_data= [
            [cKDTree(rot_frame) for rot_frame in filter_sim(sim_frame, step_size_rad, reciprocal_radius)]
            for sim_frame in simulated
        ]
    else:
        raise ValueError(f"Unsupported method: {method}")

    # array to store final results
    n_array = []

    # Loop through experimental vectors
    for exp_frame in tqdm(experimental):
        # convert to 3D
        exp3d = vector_to_3D(exp_frame, reciprocal_radius)
        # mirror over xz-plane
        exp3d_mirror = exp3d * np.array([1,-1,1])

        # array to store results per iteration of shape (n_best, 4)
        iteration_results = []
        
        for sim_idx, sim_tree_rotated in enumerate(precomputed_data):
            best_score, best_rotation, mirror = np.inf, 0.0, 0

            if method == "sum":
                best_score, best_rotation, mirror = sum_score(
                    exp3d, exp3d_mirror, sim_tree_rotated, step_size_rad)
            else: 
                raise ValueError(f"Unsupported method: {method}")
            iteration_results.append((sim_idx, best_score, best_rotation, mirror))


        # append results and sort by ascending score
        n_array.append(heapq.nsmallest(n_best, iteration_results, key=lambda x: x[1]))
            
    # returns nx4 array of shape (len(experimental), n_best, 4)
    return np.stack(n_array)


#NOTE: Dette under er for gøy
def process_exp_frame(
    exp3d: np.ndarray,
    exp3d_mirror: np.ndarray,
    sim_data,
    step_size_rad: float,
    n_best: int,
    reciprocal_radius: float,
    method: str,
    **kwargs
):
    iteration_results = []

    for sim_idx, sim_tree_rotated in enumerate(sim_data):
        if method == "sum":
            best_score, best_rotation, mirror = sum_score(
                exp3d, exp3d_mirror, sim_tree_rotated, step_size_rad
            )
            iteration_results.append((sim_idx, best_score, best_rotation, mirror))

    return heapq.nsmallest(n_best, iteration_results, key=lambda x: x[1])

             
def vm_faf(
    experimental: np.ndarray,
    simulated: np.ndarray,
    step_size: float, 
    reciprocal_radius: float,
    n_best: int,
    method: str = "sum",
    n_jobs: int = -1,
    **kwargs
) -> np.ndarray:
    """
    Docstring
    """
    
    # Convert input degrees to radians
    step_size_rad = np.deg2rad(step_size)
    if method == "sum" or method == "kd":
        # Precompute KD-trees for rotated simulated frames
        precomputed_data= [
            [cKDTree(rot_frame) for rot_frame in filter_sim(sim_frame, step_size_rad, reciprocal_radius)]
            for sim_frame in simulated
        ]
    else:
        raise ValueError(f"Unsupported method {method}")

    # precompute experimental projections and mirrors
    exp3d_all = [vector_to_3D(exp_vec, reciprocal_radius) for exp_vec in experimental]
    exp3d_mirror_all = [exp_vec * np.array([1,-1,1], dtype=np.float32) for exp_vec in exp3d_all]

    results = Parallel(n_jobs=n_jobs) (
        delayed(process_exp_frame) (
            exp3d=exp3d_all[idx],
            exp3d_mirror=exp3d_mirror_all[idx],
            sim_data=precomputed_data,
            step_size_rad=step_size_rad,
            n_best=n_best,
            reciprocal_radius=reciprocal_radius,
            method=method,
            **kwargs
        ) for idx in tqdm(range(len(experimental)))
    )
    return np.stack(results)
 
