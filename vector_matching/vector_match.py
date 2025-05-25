import heapq
from hyperspy.component import export_to_dictionary
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from joblib import Parallel, delayed
import vm_utils

def _sum_score(
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

def _sum_score_weighted(
    exp3d: np.ndarray, 
    exp3d_mirror: np.ndarray, 
    sim_trees,
    step_size_rad: float,
    distance_bound: float = 0.05
) -> tuple:
    """
    Docstring. Dette er score B, det er den beste <3
    """
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

def _validate_dimensions(experimental: np.ndarray, simulated:np.ndarray):
    # validate experimental dimensions
    if isinstance(experimental, np.ndarray) and experimental.ndim >= 2:
        exp_dim = experimental.shape[-1]
    elif isinstance(experimental, (list, np.ndarray)) and isinstance(experimental[0], np.ndarray):
        exp_dim = experimental[0].shape[-1]
    else:
        raise ValueError("Unsupported dimension for 'experimental' input.")

    # validate simulated dimensions
    if isinstance(simulated, np.ndarray) and simulated.ndim >= 2:
        sim_dim = simulated.shape[-1]
    elif isinstance(simulated, (list, np.ndarray)) and isinstance(simulated[0], np.ndarray):
        sim_dim = simulated[0].shape[-1]
    else:
        raise ValueError("Unsupported dimension for 'simulated' input.")

    # validate equal dimensions
    if exp_dim != sim_dim:
        raise ValueError(
            f"Mismatched input dimensions: experimental = {exp_dim}, simulated = {sim_dim}"
        )

    if exp_dim not in {2,3}:
        raise ValueError(
            f"Unsupported input dimension: {exp_dim}D. Only 2D (polar) or 3D (polar with intensities) supported."
        )
    return exp_dim


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
    dtype=np.float64
) -> np.ndarray:
    """
    Docstring
    """
    
    # check correct dimensions of experimental and simulated
    dimension = _validate_dimensions(experimental, simulated) 

    # check for valid methods
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
    # check for intensity dimension
    if dimension == 3:
        # slice out intensites
        exp_intensities = [frame[:, 3] for frame in experimental]   # do it like this cause its inhomogeneous
        sim_intensities = [frame[:, 3] for frame in simulated]  
        exp3d_all = [vm_utils.vector_to_3D(exp_vec[:,:3], reciprocal_radius, dtype) for exp_vec in experimental]
    else:  
        exp3d_all = [vm_utils.vector_to_3D(exp_vec, reciprocal_radius,dtype) for exp_vec in experimental]
    exp3d_mirror_all = [exp_vec * np.array([1,-1,1], dtype=dtype) for exp_vec in exp3d_all]

    if fast:
        # parallelised method, very RAM demanding
        n_array = Parallel(n_jobs=n_jobs) (
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
        # slower method, but more light on RAM
        for idx in tqdm(range(len(experimental))):

            n_array.append(process_frames(
                exp3d_all[idx], exp3d_mirror_all[idx], precomputed_data, step_size_rad, n_best,
                method, distance_bound
            ))
        # returns nx4 array of shape (len(experimental), n_best, 4)
        return np.stack(n_array)

def _get_score_intensity(
    exp3d,
    exp_tree,
    exp_intensities,
    sim_kdtree,
    sim_coords,
    sim_intensities,
    n_total,
    distance_bound,
    intensity_weight,
    intensity_norm_factor
):
    """
    Helper function for score_intensity
    """
    dist_exp_to_sim, indices_sim_for_exp = sim_kdtree.query(exp3d, distance_upper_bound=distance_bound)
    dist_sim_to_exp, indices_exp_for_sim = exp_tree.query(sim_coords, distance_upper_bound=distance_bound)

    point_scores_exp_to_sim = np.full(len(exp3d), distance_bound, dtype=float)
    matched_exp_mask = np.isfinite(dist_exp_to_sim) # mask for matched points

    if np.any(matched_exp_mask):
        dist_matched_exp = dist_exp_to_sim[matched_exp_mask]
        current_score_exp = dist_matched_exp    # only considering distance in this step

        # Add intensity if present
        exp_intensities_matched = exp_intensities[matched_exp_mask]
        sim_indices_matched = indices_sim_for_exp[matched_exp_mask]
        sim_intensities_neighbour_matched = sim_intensities[sim_indices_matched]

        intensity_diff = np.abs(exp_intensities_matched - sim_intensities_neighbour_matched)

        intensity_penalty = intensity_weight * (intensity_diff / intensity_norm_factor)

        current_score_exp += intensity_penalty

        point_scores_exp_to_sim[matched_exp_mask] = current_score_exp

    sum_scores_exp_to_sim = np.sum(point_scores_exp_to_sim)

    # now we do the same process but in the other direction
    point_scores_sim_to_exp = np.full(len(sim_coords), distance_bound, dtype=float)
    matched_sim_mask = np.isfinite(dist_sim_to_exp)

    if np.any(matched_sim_mask):
        dist_matched_sim = dist_sim_to_exp[matched_sim_mask]
        current_score_sim = dist_matched_sim

        sim_intensities_matched = sim_intensities[matched_sim_mask]
        exp_indices_matched = indices_exp_for_sim[matched_sim_mask]
        exp_intensities_neighbour_matched = exp_intensities[exp_indices_matched]

        intensity_diff_sim = np.abs(sim_intensities_matched - exp_intensities_neighbour_matched)

        intensity_penalty_sim = intensity_weight * (intensity_diff_sim / intensity_norm_factor)

        current_score_sim += intensity_penalty_sim

        point_scores_sim_to_exp[matched_sim_mask] = current_score_sim

    sum_scores_sim_to_exp = np.sum(point_scores_sim_to_exp)

    return (sum_scores_exp_to_sim + sum_scores_sim_to_exp) / n_total



def score_intensity(
    exp3d,
    exp3d_mirror,
    sim_data_items,
    exp_intensities,
    step_size_rad,
    distance_bound: float = 0.05,
    intensity_norm_factor = 1,
    intensity_weight = 1
):

    best_score, best_rotation, mirror = np.inf, 0.0, 0

    exp_tree = cKDTree(exp3d)
    exp_tree_mirror = cKDTree(exp3d_mirror)

    for rot_idx, sim_data_item in enumerate(sim_data_items):
        sim_kdtree = sim_data_item['kdtree']
        sim_coords = sim_data_item['coordinates']
        sim_intensities = sim_data_item['intensities']

        # skip empty frames
        if sim_kdtree is None or sim_coords.shape[0] == 0:
            continue

        n_total = len(exp3d) + len(sim_coords)

        score_original = _get_score_intensity(
            exp3d, exp_tree, exp_intensities, sim_kdtree, sim_coords, sim_intensities,
            n_total,distance_bound, intensity_weight, intensity_norm_factor
        )

        score_mirror = _get_score_intensity(
            exp3d_mirror, exp_tree_mirror, exp_intensities, sim_kdtree, sim_coords, sim_intensities,
            n_total,distance_bound, intensity_weight, intensity_norm_factor
        )

        scores = [
            (score_original, 1),
            (score_mirror, -1)
        ]

        for score, mirror_flag in scores:
            if score < best_score:
                best_score = score
                mirror = mirror_flag
                best_rotation = vm_utils.wrap_degrees(rot_idx * step_size_rad, mirror)

    return best_score, best_rotation, mirror




def vector_match_intensity(
    experimental,
    simulated,
    step_size,
    reciprocal_radius,
    n_best,
    method,
    n_jobs,
    fast,
    distance_bound,
    dtype=np.float64
):
    """
    Temporary main function for score_intensity
    """

    step_size_rad = np.deg2rad(step_size)
    precomputed_sim_rotations = []

    for sim_frame in simulated:
        
        # inner list to store processed frames
        processed_sim_frames = []

        for rot_frame in vm_utils.filter_sim(sim_frame, step_size_rad, reciprocal_radius, dtype):
            # create dictionary
            current_rot_frame_data = {
                'kdtree': None,
                'coordinates': np.empty((0,3)),
                'intensities': None,
                'is_valid': False
            }

            coords = rot_frame[:, :3]
            current_rot_frame_data['kdtree'] = cKDTree(coords)
            current_rot_frame_data['coordinates'] = coords
            
            # check for intensity dimension
            if rot_frame.shape[1] == 4:
                current_rot_frame_data['intensities'] = rot_frame[:,3]

            processed_sim_frames.append(current_rot_frame_data)

    precomputed_sim_rotations.append(precomputed_sim_rotations)

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
            best_score, best_rotation, mirror = _sum_score(
                exp3d, exp3d_mirror, sim_tree_rotated, step_size_rad
            )
        elif method == "sum_score_weighted":
            best_score, best_rotation, mirror = _sum_score_weighted(
                exp3d, exp3d_mirror, sim_tree_rotated, step_size_rad, distance_bound
            )
        iteration_results.append((sim_idx, best_score, best_rotation, mirror))

    return heapq.nsmallest(n_best, iteration_results, key=lambda x: x[1])
