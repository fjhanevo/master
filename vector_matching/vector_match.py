import heapq
import numpy as np
from scipy.spatial import cKDTree
from numba import njit
from tqdm import tqdm
from joblib import Parallel, delayed
import vm_utils

def sum_score(
    exp3d: np.ndarray, 
    exp3d_mirror: np.ndarray, 
    sim_trees: list,
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

def _get_sum_score_weighted(
    exp3d,
    exp_tree,
    sim_tree,
    sim_points,
    distance_bound
) -> float:
    # calculate nn distances both ways
    dist_exp_to_sim, _ = sim_tree.query(exp3d, distance_upper_bound=distance_bound)
    dist_sim_to_exp, _ = exp_tree.query(sim_points, distance_upper_bound=distance_bound)

    # penalise unmatched points by treating their distance as distance_bound
    dist_exp_to_sim_penalised = np.copy(dist_exp_to_sim)
    dist_exp_to_sim_penalised[np.isinf(dist_exp_to_sim_penalised)] = distance_bound
    dist_sim_to_exp_penalised = np.copy(dist_sim_to_exp)
    dist_sim_to_exp_penalised[np.isinf(dist_sim_to_exp_penalised)] = distance_bound

    score_exp = np.sum(dist_exp_to_sim_penalised)
    score_sim = np.sum(dist_sim_to_exp_penalised)

    return score_exp + score_sim

def sum_score_weighted(
    exp3d: np.ndarray, 
    exp3d_mirror: np.ndarray, 
    sim_trees: list,
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

        score_original = _get_sum_score_weighted(
            exp3d, exp_tree, sim_tree, sim_points, distance_bound
        )

        score_mirror = _get_sum_score_weighted(
            exp3d_mirror, exp_tree_mirror, sim_tree, sim_points, distance_bound
        )

                
        scores = [
            ( score_original / n_total, 1),
            (score_mirror / n_total, -1),
        ]
        # check score and keep only the best score for each sim_frame
        for score, mirror_flag in scores:
            if score < best_score:
                best_score = score
                ang = rot_idx * step_size_rad
                mirror = mirror_flag
                best_rotation = vm_utils.wrap_degrees(ang, mirror)
    return best_score, best_rotation, mirror

@njit
def _get_score_ang(
    exp_vecs: np.ndarray,
    sim_vecs: np.ndarray,
    ang_thresh_rad:float,
    n_total: float
): 

    matched_angles = []
    unmatched_exp = 0.0

    # for njit
    sim_vecs_cont = np.ascontiguousarray(sim_vecs)
    exp_vecs_cont = np.ascontiguousarray(exp_vecs)
    # experimental points 
    for i in range(exp_vecs.shape[0]):
        exp_vec = exp_vecs_cont[i, :]

        # dot product for one experimental vector againt all simulated vectors
        dots = np.clip(sim_vecs @ exp_vec, -1.0, 1.0)
        angles = np.arccos(dots)
        min_angle = np.min(angles)  # only keep the smallest angle
        if min_angle < ang_thresh_rad:
            matched_angles.append(min_angle)
        else:
            # penalise unmatched points by ang_thresh_rad
            unmatched_exp += ang_thresh_rad 

    # now the other direction
    unmatched_sim = 0.0
    for i in range(sim_vecs.shape[0]):
        # for njit
        sim_vec = sim_vecs_cont[i, :]

        # dot product for one simulated vector against all experimental vectors
        dots = np.clip(exp_vecs @ sim_vec, -1.0, 1.0)
        angles = np.arccos(dots)
        min_angle = np.min(angles)
        if min_angle >= ang_thresh_rad:
            unmatched_sim += ang_thresh_rad

    penalty = unmatched_exp + unmatched_sim
    if len(matched_angles) > 0:
        # get the mean score
        score = sum(matched_angles) / len(matched_angles)
    else: 
        score = np.pi   # assign a high score if no matches are found
    score += penalty
    # return normalised score
    return score / n_total

def score_ang(
    exp3d: np.ndarray,
    exp3d_mirror: np.ndarray,
    sim_data: list,
    step_size_rad: float,
    ang_thresh_rad: float,
) ->tuple:
    best_score, best_rotation, mirror = np.inf, 0.0, 0

    for rot_idx, sim_coords in enumerate(sim_data):
        sim_coord = sim_coords 

        # skip empty frames
        if sim_coords.shape[0] == 0:
            continue

        n_total = len(exp3d) + len(sim_coords)

        score_original = _get_score_ang(
            exp_vecs=exp3d,
            sim_vecs=sim_coord,
            ang_thresh_rad=ang_thresh_rad,
            n_total=n_total
        )
        score_mirror= _get_score_ang(
            exp_vecs=exp3d_mirror,
            sim_vecs=sim_coord,
            ang_thresh_rad=ang_thresh_rad,
            n_total=n_total
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

def _get_score_intensity(
    exp3d: np.ndarray,
    exp_tree: cKDTree,
    exp_intensities: np.ndarray,

    sim_kdtree: cKDTree,
    sim_coords: np.ndarray,
    sim_intensities: np.ndarray,

    n_total: float,
    distance_bound: float,
    intensity_weight: float,
):
    """
    Helper function for score_intensity
    """
    dist_exp_to_sim, indices_sim_for_exp = sim_kdtree.query(exp3d, distance_upper_bound=distance_bound)

    # initialise scores for each experimental and simulated point
    point_scores_exp = np.zeros(len(exp3d), dtype=float)
    matched_exp_mask = np.isfinite(dist_exp_to_sim) # mask for matched points
    unmatched_exp_mask = ~matched_exp_mask

    # score for matched experimental points
    if np.any(matched_exp_mask):
        exp_intensities_matched = exp_intensities[matched_exp_mask]
        sim_indices_for_matched_exp = indices_sim_for_exp[matched_exp_mask]
        sim_intensities_neighbours = sim_intensities[sim_indices_for_matched_exp]

        intensity_diff_exp = np.abs(exp_intensities_matched - sim_intensities_neighbours)

        score_matched_exp = intensity_weight * intensity_diff_exp

        point_scores_exp[matched_exp_mask] = score_matched_exp

    # score for unmatched experimental points
    if np.any(unmatched_exp_mask):
        exp_intensities_unmatched = exp_intensities[unmatched_exp_mask]
        unmatched_point_penalty_exp = intensity_weight * exp_intensities_unmatched

        point_scores_exp[unmatched_exp_mask] = unmatched_point_penalty_exp

    sum_exp_score = np.sum(point_scores_exp)

    # now we do the same for the simulated points
    dist_sim_to_exp, indices_exp_for_sim = exp_tree.query(sim_coords, distance_upper_bound=distance_bound)
    point_scores_sim = np.zeros(len(sim_coords), dtype=float)

    matched_sim_mask = np.isfinite(dist_sim_to_exp)
    unmatched_sim_mask = ~matched_sim_mask

    # score for matched simulated points
    if np.any(matched_sim_mask):
        sim_intensities_matched = sim_intensities[matched_sim_mask]
        exp_indices_for_matched_sim = indices_exp_for_sim[matched_sim_mask]
        exp_intensities_neighbours = exp_intensities[exp_indices_for_matched_sim]

        intensity_diff_sim = np.abs(sim_intensities_matched - exp_intensities_neighbours)
        score_matched_sim = intensity_weight * intensity_diff_sim 
        point_scores_sim[matched_sim_mask] = score_matched_sim

    # score for unmatched simulated points
    if np.any(unmatched_sim_mask):
        sim_intensities_unmatched = sim_intensities[unmatched_sim_mask]
        unmatched_point_penalty_sim = intensity_weight * sim_intensities_unmatched 
        point_scores_sim[unmatched_sim_mask] = unmatched_point_penalty_sim

    sum_sim_score = np.sum(point_scores_sim)

    return (sum_exp_score + sum_sim_score) / n_total

    

def score_intensity(
    exp3d: np.ndarray,
    exp3d_mirror: np.ndarray,
    exp_intensities: np.ndarray,
    sim_data_items: list,
    step_size_rad: float,
    distance_bound: float = 0.05,
    intensity_weight: float = 1,
    exp_max_intensity: float = 1.0,
    sim_max_intensity: float = 1.0,
) -> tuple:

    best_score, best_rotation, mirror = np.inf, 0.0, 0

    exp_tree = cKDTree(exp3d)
    exp_tree_mirror = cKDTree(exp3d_mirror)

    # noralise experimental intensities to range 0 to 1 
    exp_intensities_normalised = np.clip(exp_intensities / exp_max_intensity, 0.0, 1.0)

    for rot_idx, sim_data_item in enumerate(sim_data_items):
        sim_kdtree = sim_data_item['kdtree']
        sim_coords = sim_data_item['coordinates']
        sim_intensities = sim_data_item['intensities']

        # skip empty frames
        if sim_kdtree is None or sim_coords.shape[0] == 0:
            continue

        # normalise simulated intensities (sets intensities to range 0 to 1)
        sim_intensities_normalised = np.clip(sim_intensities / sim_max_intensity, 0.0, 1.0)

        n_total = len(exp3d) + len(sim_coords)

        score_original = _get_score_intensity(
            exp3d=exp3d,
            exp_tree=exp_tree,
            exp_intensities=exp_intensities_normalised,

            sim_kdtree=sim_kdtree,
            sim_coords=sim_coords,
            sim_intensities=sim_intensities_normalised,

            n_total=n_total,
            distance_bound=distance_bound,
            intensity_weight=intensity_weight
        )

        score_mirror = _get_score_intensity(
            exp3d=exp3d_mirror,
            exp_tree=exp_tree_mirror,
            exp_intensities=exp_intensities_normalised,

            sim_kdtree=sim_kdtree,
            sim_coords=sim_coords,
            sim_intensities=sim_intensities_normalised,

            n_total=n_total,
            distance_bound=distance_bound,
            intensity_weight=intensity_weight
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

def _precompute_sim_data(
    simulated: np.ndarray,
    step_size_rad: float,
    reciprocal_radius: float,
    dtype=np.float64
) -> list:
    """
    Helper function to precompute data for intensity method
    """

    precomputed_sim_rotations = []

    for sim_frame in simulated:
        
        # inner list to store processed frames
        processed_sim_frames = []

        for rot_frame in vm_utils.filter_sim(sim_frame, step_size_rad, reciprocal_radius, dtype):
            # create dictionary
            current_rot_frame_data = {
                'kdtree': None,
                'coordinates': np.empty((0,3), dtype=dtype),
                'intensities': None,
                'is_valid': False
            }

            # print(rot_frame.shape[1])
            coords = rot_frame[:, :3]
            current_rot_frame_data['kdtree'] = cKDTree(coords)
            current_rot_frame_data['coordinates'] = coords
            
            # check for intensity dimension
            if rot_frame.shape[1] == 4:
                current_rot_frame_data['intensities'] = rot_frame[:,3].astype(dtype=dtype)

            processed_sim_frames.append(current_rot_frame_data)

        precomputed_sim_rotations.append(processed_sim_frames)

    return precomputed_sim_rotations

def process_frames(
    exp3d: np.ndarray,
    exp3d_mirror: np.ndarray,
    exp_intensities: np.ndarray,
    sim_precomputed: list,
    step_size_rad: float,
    n_best: int,
    method: str="sum_score",
    **kwargs
) -> list:
    iteration_results = []

    for sim_idx, sim_data in enumerate(sim_precomputed):
        if method == "sum_score":
            sim_kdtrees = [item['kdtree'] for item in sim_data if item.get('kdtree')]
            best_score, best_rotation, mirror = sum_score(
                exp3d=exp3d,
                exp3d_mirror=exp3d_mirror,
                sim_trees=sim_kdtrees,
                step_size_rad=step_size_rad
            )

        elif method == "sum_score_weighted":
            sim_kdtrees = [item['kdtree'] for item in sim_data if item.get('kdtree')]
            best_score, best_rotation, mirror = sum_score_weighted(
                exp3d=exp3d,
                exp3d_mirror=exp3d_mirror,
                sim_trees=sim_kdtrees,
                step_size_rad=step_size_rad,
                distance_bound=kwargs.get("distance_bound", 0.05)

            )

        elif method == "score_ang":
            sim_coords = [item['coordinates'] for item in sim_data if item.get('coordinates') is not None]
            sim_coords = sim_coords
            best_score, best_rotation, mirror = score_ang(
                exp3d=exp3d,
                exp3d_mirror=exp3d_mirror,
                sim_data=sim_coords,
                step_size_rad=step_size_rad,
                ang_thresh_rad=kwargs.get("ang_thresh_rad", 0.05)
            )

        elif method == "score_intensity":
            best_score, best_rotation, mirror = score_intensity(
                exp3d=exp3d,
                exp3d_mirror=exp3d_mirror,
                exp_intensities=exp_intensities,
                sim_data_items=sim_data,
                step_size_rad=step_size_rad,
                distance_bound=kwargs.get("distance_bound", 0.05),
                intensity_weight=kwargs.get("intensity_weight", 1.0),
                exp_max_intensity=kwargs.get("exp_max_intensity", 1.0),
                sim_max_intensity=kwargs.get("sim_max_intensity", 1.0)
            )
        else:
            raise ValueError(f"Method {method} not supported.")
    
        iteration_results.append((sim_idx, best_score, best_rotation, mirror))

    # sort after ascending score
    return heapq.nsmallest(n_best, iteration_results, key=lambda x : x[1])

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
    intensity_weight: float = 1.0,
    intensity_norm_factor: float = 1.0,
    ang_thresh_rad: float = 0.05,
    dtype=np.float64
) -> np.ndarray:
    """
    Docstring
    """
    
    # check correct dimensions of experimental and simulated
    dimension = _validate_dimensions(experimental, simulated) 

    # check for valid methods
    valid_methods = {"sum_score", "sum_score_weighted", "score_ang" ,"score_intensity"}
    if method not in valid_methods:
        raise ValueError(f"Unsupported method: {method}. Valid options: {valid_methods}")

    if method == "score_ang" and fast:
        raise ValueError(f"Method: {method} does not support fast == {fast}, set fast == False")

    if dimension == 3 and method == "score_intensity":
        # 2D polar with intensity
        exp3d_all = [vm_utils.vector_to_3D(exp_vec[:,:2], reciprocal_radius,dtype) for exp_vec in experimental]
        exp_intensities = [frame[:, 2] for frame in experimental]
    elif dimension == 2:
        # 2D polar only
        exp3d_all = [vm_utils.vector_to_3D(exp_vec, reciprocal_radius,dtype) for exp_vec in experimental]
        exp_intensities = [np.zeros(len(frame)) for frame in experimental] # set to zero as we're not dealing with it
    else:
        raise ValueError(f"Wrong dimension: {dimension}D for method: {method}")
    # mirror version 
    exp3d_mirror_all = [exp_vec * np.array([1,-1,1], dtype=dtype) for exp_vec in exp3d_all]


    # Convert input degrees to radians
    step_size_rad = np.deg2rad(step_size)
    # Precompute KD-trees for rotated simulated frames
    precomputed_data = _precompute_sim_data(simulated, step_size_rad, reciprocal_radius, dtype)

    kwargs = {
        "distance_bound": distance_bound,
    }

    if method == "score_intensity":
        kwargs["intensity_weight"] = intensity_weight
        kwargs["intensity_norm_factor"] = intensity_norm_factor
        # compute global max intensities for normalising intensities
        exp_avg_intensity = [np.mean(frame[:, -1]) for frame in experimental]
        sim_avg_intensity = np.mean(simulated[:, :, -1], axis=1)
        kwargs["exp_intensity_max"] = np.max(exp_avg_intensity)
        kwargs["sim_intensity_max"] = np.max(sim_avg_intensity)
    elif method == "score_ang":
        kwargs["ang_thresh_rad"] = ang_thresh_rad

    # array to store final results
    n_array = []

    if fast:
        # parallelised method, very RAM demanding
        n_array = Parallel(n_jobs=n_jobs) (
        delayed(process_frames) (
            exp3d=exp3d_all[idx],
            exp3d_mirror=exp3d_mirror_all[idx],
            exp_intensities=exp_intensities[idx],
            sim_precomputed=precomputed_data,
            step_size_rad=step_size_rad,
            n_best=n_best,
            method=method,
            **kwargs
        ) for idx in tqdm(range(len(experimental)))
    )

    else:
        # slower method, but more light on RAM
        for idx in tqdm(range(len(experimental))):
            n_array.append(process_frames(
                exp3d=exp3d_all[idx],
                exp3d_mirror=exp3d_mirror_all[idx],
                exp_intensities=exp_intensities[idx],
                sim_precomputed=precomputed_data,
                step_size_rad=step_size_rad,
                n_best=n_best,
                method=method,
                **kwargs
            ))
        # returns nx4 array of shape (len(experimental), n_best, 4)
    return np.stack(n_array)


