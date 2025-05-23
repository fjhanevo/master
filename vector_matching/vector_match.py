import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import heapq
from joblib import Parallel, delayed
import vm_utils

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
                best_rotation = vm_utils.wrap_degrees(ang, mirror)

    return best_score, best_rotation, mirror

def sum_score_weighted(
    exp3d: np.ndarray, 
    exp3d_mirror: np.ndarray, 
    sim_trees,
    step_size_rad: float,
    distance_bound: float = 0.05
) -> tuple:
    best_score, best_rotation, mirror = np.inf, 0.0, 0

    # experimental trees
    exp_tree = cKDTree(exp3d)
    exp_tree_mirror = cKDTree(exp3d_mirror)

    for rot_idx, sim_tree in enumerate(sim_trees):
        # get total points for normalisation
        sim_points = sim_tree.data
        n_total = len(exp3d) + len(sim_points)

        # skip if 0 vectors are found for HMS safety guidelines
        if n_total == 0:
            continue


        # calculate nn distances both ways
        dist_exp_to_sim, _ = sim_tree.query(exp3d, distance_upper_bound=distance_bound)
        dist_sim_to_exp, _ = exp_tree.query(sim_points, distance_upper_bound=distance_bound)

        # mirrored version
        dist_exp_to_sim_m, _ = sim_tree.query(exp3d_mirror,distance_upper_bound=distance_bound)
        dist_sim_to_exp_m, _ = exp_tree_mirror.query(sim_points,distance_upper_bound=distance_bound)

        # penalise unmatched points by treating their distance as distance_bound
        dist_exp_to_sim_penalised = np.copy(dist_exp_to_sim)
        dist_exp_to_sim_penalised[np.isinf(dist_exp_to_sim_penalised)] = distance_bound
        dist_sim_to_exp_penalised = np.copy(dist_sim_to_exp)
        dist_sim_to_exp_penalised[np.isinf(dist_sim_to_exp_penalised)] = distance_bound

        # mirrored version
        dist_exp_to_sim_penalised_m = np.copy(dist_exp_to_sim_m)
        dist_exp_to_sim_penalised_m[np.isinf(dist_exp_to_sim_penalised_m)] = distance_bound
        dist_sim_to_exp_penalised_m = np.copy(dist_sim_to_exp_m)
        dist_sim_to_exp_penalised_m[np.isinf(dist_sim_to_exp_penalised_m)] = distance_bound

        # unmirrored score
        score_exp = np.sum(dist_exp_to_sim_penalised)
        score_sim = np.sum(dist_sim_to_exp_penalised)

        # mirror score
        score_exp_m = np.sum(dist_exp_to_sim_penalised_m)
        score_sim_m = np.sum(dist_sim_to_exp_penalised_m)
        
        scores = [
            ((score_exp + score_sim) / n_total, 1),
            ((score_exp_m + score_sim_m) / n_total, -1),
        ]
        # check score and keep only the best score for each sim_frame
        for score, mirror_flag in scores:
            if score < best_score:
                best_score = score
                ang = rot_idx * step_size_rad
                mirror = mirror_flag
                best_rotation = vm_utils.wrap_degrees(ang, mirror)
    return best_score, best_rotation, mirror


def vector_match(
    experimental: np.ndarray,
    simulated: np.ndarray,
    step_size: float, 
    reciprocal_radius: float,
    n_best: int,
    method: str = "sum",
    fast: bool = False,
    n_jobs: int = -1,
    distance_bound: float = 0.05,
    dtype=np.float32
) -> np.ndarray:
    """
    Docstring
    """
    
    valid_methods = {"sum_score", "sum_score_weighted"}
    if method not in valid_methods:
        raise ValueError(f"Unsupported method: {method}. Valid options: {valid_methods}")

    # Convert input degrees to radians
    step_size_rad = np.deg2rad(step_size)
    # Precompute KD-trees for rotated simulated frames
    precomputed_data= [
        [cKDTree(rot_frame[:, :3]) for rot_frame in vm_utils.filter_sim(sim_frame, step_size_rad, reciprocal_radius,dtype)]
        for sim_frame in simulated
    ]

    # array to store final results
    n_array = []
    # Pre-compute exp3d and its mirror
    exp3d_all = [vm_utils.vector_to_3D(exp_vec, reciprocal_radius,dtype) for exp_vec in experimental]
    exp3d_mirror_all = [exp_vec * np.array([1,-1,1], dtype=dtype) for exp_vec in exp3d_all]

    # Loop through experimental vectors
    if fast:
        n_array= Parallel(n_jobs=n_jobs) (
        delayed(process_frames) (
            exp3d=exp3d_all[idx],
            exp3d_mirror=exp3d_mirror_all[idx],
            sim_data=precomputed_data,
            step_size_rad=step_size_rad,
            n_best=n_best,
            method=method,
            distance_bound=distance_bound
        ) for idx in tqdm(range(len(experimental)))
    )
        return np.stack(n_array)
    else:
        for idx in tqdm(range(len(experimental))):

            n_array.append(process_frames(
                exp3d_all[idx], exp3d_mirror_all[idx], precomputed_data, step_size_rad, n_best,
                method, distance_bound
            ))
        # returns nx4 array of shape (len(experimental), n_best, 4)
        return np.stack(n_array)


def process_frames(
    exp3d: np.ndarray,
    exp3d_mirror: np.ndarray,
    sim_data,
    step_size_rad: float,
    n_best: int,
    method: str="sum",
    distance_bound: float = 0.05,
):
    iteration_results = []

    for sim_idx, sim_tree_rotated in enumerate(sim_data):
        best_score, best_rotation, mirror = np.inf, 0.0, 0
        if method == "sum":
            best_score, best_rotation, mirror = sum_score(
                exp3d, exp3d_mirror, sim_tree_rotated, step_size_rad
            )
        elif method == "sum_score_weighted":
            best_score, best_rotation, mirror = sum_score_weighted(
                exp3d, exp3d_mirror, sim_tree_rotated, step_size_rad, distance_bound
            )
        iteration_results.append((sim_idx, best_score, best_rotation, mirror))

    return heapq.nsmallest(n_best, iteration_results, key=lambda x: x[1])

             
