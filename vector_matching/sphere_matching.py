import pyxem as pxm
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from numba import njit, prange
import heapq

"""
Fil for sphere matching!:)
"""

def fast_polar(cart_vec: np.ndarray) -> np.ndarray:
    """
    Polar transforms 2D cartesian vectors to 2D polar vectors
    using pyxem, wow!
    Params:
    ---------
        cart_vec: np.ndarray:
            2D cartesian vector
    Returns:
    ---------
        s.data: np.ndarray:
            2D polar vector
    """
    s = pxm.signals.DiffractionVectors(cart_vec)
    s = s.to_polar()
    return s.data

def vector_to_3D(vector: np.ndarray,reciprocal_radius: float) -> np.ndarray:
    """
    Takes in a 2D polar vector and converts it to 
    a 3D sphere.
    Params:
    ---------
        vector: np.ndarray
            2D polar vector (r,theta)
        reciprocal_radius: float
            reciprocal space radius to account for
    Returns:
    ---------
        vector3d: np.ndarray
            3D vector coordinates
    """
    # get r coords
    r = vector[...,0]
    # get theta coords
    theta = vector[...,1]

    l = 2*np.arctan(r/(2*reciprocal_radius))

    # 3D coords
    x = np.sin(l)*np.cos(theta)
    y = np.sin(l)*np.sin(theta)
    z = np.cos(l)

    vector3d = np.stack([x,y,z],axis=-1)

    return vector3d

def apply_z_rotation(vector: np.ndarray,theta: float) -> np.ndarray:
    """
    It just rotates the sphere around the z-axis with a given angle.
    Params:
    ---------
        vector: np.ndarray
            input 3D vector
        theta: float
            angle to rotate vector for in radians 
    Returns:
    ---------
        np.ndarray
            the dot product between rot matrix and input vector
    """

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot = np.array([[cos_theta, -sin_theta, 0],
                    [sin_theta, cos_theta, 0],
                    [0,0,1]])
    return vector @ rot.T

def full_z_rotation(vector: np.ndarray, step_size: float) -> np.ndarray:
    """
    Helper function to add a rotation dimension.
    Params:
    ---------
        vector: np.ndarray
            3D input vector
        step_size: float
            angular increment
    Returns:
    ---------
        np.ndarray:
        fully rotated vector around the z-axis.
    """
    angles = np.arange(0,2*np.pi,step_size).tolist()
    return np.array([apply_z_rotation(vector, theta) for theta in angles])

def filter_sim(sim: np.ndarray, step_size: float, reciprocal_radius: float) -> np.ndarray:
    """
    Helper function for vector_match() to filter out zeros
    from sim because its homo, and now its in-homo.
    Params:
    ---------
        sim: np.ndarray
            simulated 3D vector
        step_size: float
            angular increment 
        reciprocal_radius: float
            reciprocal space radius to account for
    Returns:
    ---------
        full_z_rotation(): np.ndarray
            rotates the vector fully around the z-axis for given
            step_size, adds a new dimension in the dataset,
            from (N,3) to (M,N,3).
    """
    sim_filtered = sim[~np.all(sim == 0, axis=1)]

    sim_filtered_3d = vector_to_3D(sim_filtered, reciprocal_radius)

    return full_z_rotation(sim_filtered_3d,step_size)

@njit
def wrap_degrees(angle_rad: float, mirror: int) -> int:
    """
    Converts radian rotation into degrees, re-align reference axis
    with Pyxems conventions. Wraps it to the range (-180, 180) deg, 
    with counter-clockwise rotation
    Params:
    ---------
        angle_rad: float
            the in-plane angle found from VM in radians
        mirror: int
            mirror factor from VM, either 1 or -1 depending on if the 
            pattern was mirrored. Used to reverse rotation if mirrored
            (mirror == -1)
    Returns:
    ---------
        angle_deg: int 
            the in-plane rotation in degrees and changed to match Pyxems conventions.
            if mirrored, returns the reverse rotation
            if not mirrored, add 180 degs to match pyxem. 
    """
    angle_deg = int(np.rad2deg((angle_rad-np.pi/2)  % (2*np.pi)))
    if angle_deg > 180:
        angle_deg -= 360
    if mirror < 0:
        # reverse rotation when mirrored
        angle_deg = -angle_deg
        return angle_deg
    # Add 180 deg to match pyxem conventions
    else: 
        return angle_deg + 180

def vector_match_kd(
    experimental: np.ndarray, 
    simulated: np.ndarray, 
    step_size: float, 
    reciprocal_radius: float, 
    n_best: int,
    distance_bound: float = 0.05,
) -> np.ndarray:
    """
    Vector matching method based on KD-trees nearest-neighbour distance matching.
    Calculates nn-distances between points in a given threshold, penalises unmatched
    experimental and simulated points. 
    Params:
    ---------
        experimental: np.ndarray:
            Experimental 2D vector points of shape (N,2)
        simulated: np.ndarray: 
            Simulated 2D vector points of shape (M,P,2)
        step_size: float:
            The angular step size increment used to rotate the simulated 3D spheres
        reciprocal_radius: float: 
            The reciprocal radius used to project vector points onto sphere with this radius
        n_best: int: 
            Number of n best matches to keep
        distance_bound: float: 
            Distance threshold to account for when matching points. If a point is 
            within the threshold of another point, their nearest-neighbour distance is calculated.

    Returns:
    ---------
        n_array: np.ndarray:
            A nx4-shaped array of [frame, score, in-plane, mirror-factor]
    """
    n_array = []

    # Convert input degrees to radians
    step_size_rad = np.deg2rad(step_size)
    # precompute KD-trees for all rotated sims
    precomputed_trees = [
        [cKDTree(rot_frame) for rot_frame in filter_sim(sim_frame, step_size_rad, reciprocal_radius)]
        for sim_frame in simulated
    ]
    # Loop through all experimental vectors
    for exp_vec in tqdm(experimental):
        # Transpose to 3D
        exp3d = np.array(vector_to_3D(exp_vec,reciprocal_radius))
        # Mirror exp3d over the XZ-plane
        exp3d_mirror = exp3d * np.array([1,-1,1])
        results = []

        # Loop through each simulated frame
        for sim_idx, trees in enumerate(precomputed_trees):
            # just reset and declare these here to stop the lsp from bitching
            best_score, best_rotation, mirror = float('inf'), 0, 0

            # Loop through each rotation of the simulated frame
            for rot_idx, sim_tree in enumerate(trees):
                # Points in current sim frame
                sim_points = sim_tree.data

                # total points
                n_total = len(exp3d) + len(sim_points)
                
                # Experimental tree
                exp_tree = cKDTree(exp3d)
                exp_tree_mirror = cKDTree(exp3d_mirror)

                # Original version
                dist_exp_to_sim, _ = sim_tree.query(exp3d, distance_upper_bound=distance_bound)
                dist_sim_to_exp, _ =exp_tree.query(sim_points, distance_upper_bound=distance_bound)

                n_unmatched_exp = np.sum(np.isinf(dist_exp_to_sim))
                n_unmatched_sim = np.sum(np.isinf(dist_sim_to_exp))
                matched_score = np.sum(dist_exp_to_sim[np.isfinite(dist_exp_to_sim)])
                # normalise score
                score = (matched_score + (n_unmatched_exp + n_unmatched_sim))/n_total

                # Mirrored version
                dist_exp_to_sim_m, _ = sim_tree.query(exp3d_mirror,distance_upper_bound=distance_bound)
                dist_sim_to_exp_m, _ = exp_tree_mirror.query(sim_points,distance_upper_bound=distance_bound)

                n_unmatched_exp_m = np.sum(np.isinf(dist_exp_to_sim_m))
                n_unmatched_sim_m = np.sum(np.isinf(dist_sim_to_exp_m))
                matched_score_m = np.sum(dist_exp_to_sim_m[np.isfinite(dist_exp_to_sim_m)])
                # normalise score
                score_mirror = (matched_score_m +  (n_unmatched_exp_m + n_unmatched_sim_m))/n_total

                # Check score and keep only best score for each sim_frame
                if score < best_score:
                    best_score = score
                    ang = rot_idx * step_size_rad 
                    mirror = 1
                    best_rotation = wrap_degrees(ang,mirror) 

                if score_mirror < best_score:
                    best_score = score_mirror
                    ang = rot_idx * step_size_rad 
                    mirror = -1
                    best_rotation = wrap_degrees(ang,mirror) 
            # Store results for each sim_frame
            # nx4-shape [frame, score, in-plane, mirror-factor]
            results.append((sim_idx, best_score, best_rotation, mirror))
        
        # Sort by ascending score and select n_best
        results = sorted(results, key = lambda x : x[1])[:n_best]
        n_array.append(np.array(results))
    # Return array of shape (len(experimental), n_best, 4)
    return np.array(n_array)

def angular_score(
    exp_vecs: np.ndarray,
    sim_vecs: np.ndarray,
    angle_thresh_rad: float = 0.05
):
    matched_angles = []
    unmatched_exp = 0
    for exp_vec in exp_vecs:
        dots = np.clip(sim_vecs @ exp_vec, -1.0, 1.0)
        angles = np.arccos(dots)
        min_angle = np.min(angles)
        if min_angle < angle_thresh_rad:
            matched_angles.append(min_angle)
        else:
            unmatched_exp += 1

    unmatched_sim = 0
    for sim_vec in sim_vecs:
        dots = np.clip(exp_vecs @ sim_vec, -1.0, 1.0)
        angles = np.arccos(dots)
        min_angle = np.min(angles)
        if min_angle >= angle_thresh_rad:
            unmatched_sim += 1

    n_tot = len(exp_vecs) + len(sim_vecs)
    score = np.mean(matched_angles) if matched_angles else np.pi
    score += unmatched_exp + unmatched_sim
    # return normalised score
    return score / n_tot

def vector_match_ang_score(
    experimental: np.ndarray,
    simulated: np.ndarray,
    step_size: float,
    reciprocal_radius: float,
    n_best: int,
    angle_thresh_rad: float = 0.05
):
    """
    Vector matching method based on computing the angular score of neighbouring points
    within a given threshold. Penalises the score for unmatched experimental and simulated
    vector points. 
    Params:
    ---------
        experimental: np.ndarray:
            Experimental 2D vector points of shape (N,2)
        simulated: np.ndarray: 
            Simulated 2D vector points of shape (M,P,2)
        step_size: float:
            The angular step size increment used to rotate the simulated 3D spheres
        reciprocal_radius: float: 
            The reciprocal radius used to project vector points onto sphere with this radius
        n_best: int: 
            Number of n best matches to keep
        angle_thresh_rad: float: 
            Angular threshold radius to account for when matching points. If a point is 
            within the threshold of another point, their angular difference is calculated.

    Returns:
    ---------
        n_array: np.ndarray:
            A nx4-shaped array of [frame, score, in-plane, mirror-factor]

    """
    step_size_rad = np.deg2rad(step_size)

    precomputed_rotated_vectors = [
        filter_sim(sim_frame, step_size_rad, reciprocal_radius)
        for sim_frame in simulated
    ]

    n_array = []

    for exp_vec in tqdm(experimental):
        exp3d = vector_to_3D(exp_vec, reciprocal_radius)
        # Mirror exp3d over the XZ-plane
        exp3d_mirror = exp3d * np.array([1,-1,1])
        results = []

        for sim_idx, sim_rotations in enumerate(precomputed_rotated_vectors):
            best_score, best_rotation, mirror = float('inf'), 0.0, 0 

            for rot_idx, sim3d in enumerate(sim_rotations):
                score = angular_score(exp3d, sim3d, angle_thresh_rad)
                score_mirror = angular_score(exp3d_mirror, sim3d, angle_thresh_rad)

                if score < best_score:
                    best_score = score
                    ang = rot_idx * step_size_rad
                    mirror = 1
                    best_rotation = wrap_degrees(ang, mirror)

                if score_mirror < best_score:
                    best_score = score_mirror
                    ang = rot_idx * step_size_rad
                    mirror = -1
                    best_rotation = wrap_degrees(ang, mirror)

            results.append((sim_idx, best_score, best_rotation, mirror))

        # sort and keep best matches
        results = sorted(results, key=lambda x: x[1])[:n_best]
        n_array.append(np.array(results))

    return np.array(n_array)



def vector_match_sum_score(
    experimental: np.ndarray, 
    simulated: np.ndarray, 
    step_size: float, 
    reciprocal_radius: float, 
    n_best: int,
    distance_bound: float = 0.05
) -> np.ndarray:
    """
    Vector matching method based on KD-trees, takes only nearest-neighbour distances
    into account. 
    Params:
    ---------
        experimental: np.ndarray:
            Experimental 2D vector points of shape (N,2)
        simulated: np.ndarray: 
            Simulated 2D vector points of shape (M,P,2)
        step_size: float:
            The angular step size increment used to rotate the simulated 3D spheres
        reciprocal_radius: float: 
            The reciprocal radius used to project vector points onto sphere with this radius
        n_best: int: Number of n best matches to keep
        distance_bound: float: 
            Distance threshold to account for when matching points. If a point is 
            within the threshold of another point, their nearest-neighbour distance is calculated.

    Returns:
    ---------
        n_array: np.ndarray:
            A nx4-shaped array of [frame, score, in-plane, mirror-factor]
    """
    n_array = [] 
    # Convert input degrees to radians
    step_size_rad= np.deg2rad(step_size)
    # precompute KD-trees for all rotated sims
    precomputed_trees = [
        [cKDTree(rot_frame) for rot_frame in filter_sim(sim_frame, step_size_rad, reciprocal_radius)]
        for sim_frame in simulated
    ]
    # Loop through all experimental vectors
    for exp_vec in tqdm(experimental):
        # Transpose to 3D
        exp3d = vector_to_3D(exp_vec,reciprocal_radius)
        # Mirror exp3d over the XZ-plane
        exp3d_mirror = exp3d * np.array([1,-1,1])
        results = []
        # Loop through each simulated frame
        for sim_idx, trees in enumerate(precomputed_trees):
            # just reset and declare these here to stop the lsp from bitching
            best_score, best_rotation, mirror = float('inf'), 0, 0
            # Loop through each rotation of the simulated frame
            for rot_idx, sim_tree in enumerate(trees):

                sim_points = sim_tree.data

                n_total = len(exp3d) + len(sim_points)
                distances, _ = sim_tree.query(exp3d, distance_upper_bound=distance_bound)
                # low score is good
                score = np.sum(distances)/n_total
                # mirror score
                distances_mirror, _ = sim_tree.query(exp3d_mirror, distance_upper_bound=distance_bound)
                score_mirror = np.sum(distances_mirror)/n_total
                # Check score and keep only best score for each sim_frame
                if score < best_score:
                    best_score = score
                    ang = rot_idx * step_size_rad
                    mirror = 1
                    best_rotation = wrap_degrees(ang,mirror) 
                # Check mirror score
                if score_mirror < best_score:
                    best_score = score
                    ang = rot_idx * step_size_rad
                    mirror = -1
                    best_rotation = wrap_degrees(ang,mirror) 
            # Store results for each sim_frame
            # nx4-shape [frame, score, rotation, mirror-factor]
            results.append((sim_idx, best_score, best_rotation, mirror))
        # Sort by ascending score and select n_best
        results = sorted(results, key = lambda x : x[1])[:n_best]
        n_array.append(np.array(results))
    # Return array of shape (len(experimental), n_best, 4)
    return np.array(n_array)
 

@njit
def compute_best_score(
    exp3d, 
    exp3d_mirror,
    sim_trees, 
    step_size_rad,
    distance_bound,
):
    best_score, best_rotation, mirror = np.inf, 0.0, 0

    for rot_idx in range(len(sim_trees.data)):
        sim_tree_data = sim_trees[rot_idx].data

        n_total = len(exp3d) + len(sim_tree_data)

        score, score_mirror = 0.0, 0.0

        # Manual KD-trees wow!
        for i in range(len(exp3d)):
            d_min, d_min_mirror = distance_bound, distance_bound

            for j in range(len(sim_tree_data)):
                d = np.linalg.norm(exp3d[i] - sim_tree_data[j])
                d_mirror = np.linalg.norm(exp3d_mirror[i] - sim_tree_data[j])
                if d < d_min:
                    d_min = d
                if d_mirror < d_min_mirror:
                    d_min_mirror = d_mirror
            score += d_min
            score_mirror += d_min_mirror

        score /= n_total
        score_mirror /= n_total

        if score < best_score:
            best_score = score
            mirror = 1
            best_rotation = wrap_degrees(rot_idx * step_size_rad, mirror)
        if score_mirror < best_score:
            best_score = score
            mirror = 1
            best_rotation = wrap_degrees(rot_idx * step_size_rad, mirror)

    return best_score, best_rotation, mirror

def vector_match_faf(
    experimental: np.ndarray,
    simulated: np.ndarray,
    step_size: float,
    reciprocal_radius: float,
    n_best: int,
    distance_bound: float = 0.05,
) -> np.ndarray:
    """
    DOCSTRING!
    """
    n_array = []

    # Precomput-sim trees for faster runtime
    step_size_rad = np.deg2rad(step_size)
    precomputed_trees = [
        [cKDTree(rot_frame) for rot_frame in filter_sim(sim_frame, step_size_rad, reciprocal_radius)]
        for sim_frame in simulated
    ]

    # Loop through all experimental vectors
    for exp_vec in tqdm(experimental):
        # Transpose to 3D
        exp3d = vector_to_3D(exp_vec,reciprocal_radius)
        # Mirror exp3d over the XZ-plane
        exp3d_mirror = exp3d * np.array([1,-1,1])
        results = []

        for sim_idx, sim_tree in enumerate(precomputed_trees):
            best_score, best_rotation, mirror = compute_best_score(
                exp3d, exp3d_mirror, sim_tree, step_size_rad, distance_bound
            ) 
            results.append((sim_idx, best_score, best_rotation, mirror))

        n_array.append(np.array(heapq.nsmallest(n_best, results, key = lambda x: x[1])))
    return np.array(n_array)





    
def vector_match(
    experimental: np.ndarray,
    simulated: np.ndarray,
    step_size: float,
    reciprocal_radius: float,
    n_best: int,
    method: int = 0, 
    distance_bound: float = 0.05,
    angle_thresh_rad: float = 0.05
) -> np.ndarray:
    """
    Just a central function to choose method
    method == 1: vector_match_kd,
    method == 2: vector_match_ang_score,
    method == 3: vector_match_sum_score,
    """

    n_array = np.array([])
    if method == 1:
        n_array = vector_match_kd(experimental, simulated, step_size,reciprocal_radius,n_best, distance_bound)
    elif method == 2:
        n_array = vector_match_ang_score(experimental, simulated, step_size,reciprocal_radius,n_best, angle_thresh_rad)
    elif method == 3:
        n_array = vector_match_sum_score(experimental, simulated, step_size, reciprocal_radius, n_best)
    elif method == 4:
        n_array = vector_match_faf(experimental, simulated, step_size,reciprocal_radius,n_best, distance_bound)
    else:
        print("Invalid input for method: {}", method)
    return n_array
