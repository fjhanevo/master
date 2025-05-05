import pyxem as pxm
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

"""
Fil for sphere matching!:)
"""

def fast_polar(cart_vec):
    """
    Polar transforms 2D cartesian vectors to 2D polar vectors
    using pyxem, wow!
    """
    s = pxm.signals.DiffractionVectors(cart_vec)
    s = s.to_polar()
    return s.data

def vector_to_3D(vector:np.ndarray,reciprocal_radius:float) -> np.ndarray:
    """
    Takes in a 2D polar vector and converts it to 
    a 3D sphere.
    Params:
    -------
        vector: np.ndarray
            2D polar vector (r,theta)
        reciprocal_radius: float
            reciprocal space radius to account for
    Returns:
    -------
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

def apply_z_rotation(vector:np.ndarray,theta:float) -> np.ndarray:
    """
    It just rotates the sphere around the z-axis with a given angle.
    Params:
    -------
        vector: np.ndarray
            input 3D vector
        theta: float
            angle to rotate vector for in radians 
    Returns:
    -------
        np.ndarray
            the dot product between rot matrix and input vector
    """

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot = np.array([[cos_theta, -sin_theta, 0],
                    [sin_theta, cos_theta, 0],
                    [0,0,1]])
    return vector @ rot.T

#NOTE: Will remove this and deal with intensities in a different way
def z_rotation_intensity(vector:np.ndarray, theta:float) -> np.ndarray:
    # pop out intensities
    temp_vector = np.stack([vector[...,0], vector[...,1], vector[...,2]], axis=-1)
    rot_vector = np.array(apply_z_rotation(temp_vector, theta))
    # pop intensities back in
    return np.stack([rot_vector[...,0], rot_vector[...,1], rot_vector[...,2], vector[...,3]], axis=-1)

def full_z_rotation(vector:np.ndarray, step_size:float) -> np.ndarray:
    """
    Helper function to add a rotation dimension.
    Params:
    -------
        vector: np.ndarray
            3D input vector
        step_size: float
            angular increment
    Returns:
    -------
        np.ndarray:
        fully rotated vector around the z-axis.
    """
    angles = np.arange(0,2*np.pi,step_size).tolist()
    return np.array([apply_z_rotation(vector, theta) for theta in angles])

def filter_sim(sim:np.ndarray, step_size:float, reciprocal_radius:float) -> np.ndarray:
    """
    Helper function for vector_match() to filter out zeros
    from sim because its homo, and now its in-homo.
    Params:
    -------
        sim: np.ndarray
            simulated 3D vector
        step_size: float
            angular increment 
        reciprocal_radius: float
            reciprocal space radius to account for
    Returns:
    -------
        full_z_rotation(): np.ndarray
            rotates the vector fully around the z-axis for given
            step_size, adds a new dimension in the dataset,
            from (N,3) to (M,N,3).
    """
    sim_filtered = sim[~np.all(sim == 0, axis=1)]

    sim_filtered_3d = vector_to_3D(sim_filtered, reciprocal_radius)

    return full_z_rotation(sim_filtered_3d,step_size)

#NOTE: Add argument for mirror
def wrap_degrees(angle_rad):
    """
    Function to fix the 90 deg shift to match Pyxems convention.
    Wraps around if deg > 180. 
    """
    angle_deg = int(np.rad2deg((angle_rad - np.pi/2) % (2*np.pi)))
    # angle_deg = int(np.rad2deg((angle_rad) % (2*np.pi)))
    if angle_deg > 180:
        angle_deg -= 360
    # Add 180 deg to match pyxem conventions
    return angle_deg + 180

def vector_match(
    experimental: np.ndarray, 
    simulated: np.ndarray, 
    step_size: float, 
    reciprocal_radius: float, 
    n_best: int,
    distance_bound: float = 0.05,
    unmatched_penalty: float = 1.0 
) -> np.ndarray:
    """
    Oppdaterer denne senere.
    """
    n_array = []

    #TODO: Add pre-filtering step based on n_best (spot match)
    #NOTE: kommer nok ikke til å gjøre dette

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
        # Mirror exp3d over the YZ-plane
        #NOTE: mirror om y!
        exp3d_mirror = exp3d * np.array([1,-1,1])
        results = []

        # Loop through each simulated frame
        for sim_idx, trees in enumerate(precomputed_trees):
            # just reset and declare these here to stop the lsp from bitching
            best_score, best_rotation, mirror = float('inf'), 0, 0.

            # Loop through each rotation of the simulated frame
            for rot_idx, sim_tree in enumerate(trees):
                # Points in current sim frame
                sim_points = sim_tree.data

                # total points
                n_total = len(exp3d) + len(sim_points)
                
                # Experimental tree
                exp_tree = cKDTree(exp3d)
                #NOTE: HEr også !
                exp_tree_mirror = cKDTree(exp3d_mirror)

                # Original version
                dist_exp_to_sim, _ = sim_tree.query(exp3d,distance_upper_bound=distance_bound)
                dist_sim_to_exp, _ =exp_tree.query(sim_points,distance_upper_bound=distance_bound)

                n_unmatched_exp = np.sum(np.isinf(dist_exp_to_sim))
                n_unmatched_sim = np.sum(np.isinf(dist_sim_to_exp))
                matched_score = np.sum(dist_exp_to_sim[np.isfinite(dist_exp_to_sim)])
                # normalise score
                score = (matched_score + unmatched_penalty * (n_unmatched_exp + n_unmatched_sim))/n_total

                #NOTE: Mirror er fjerna for nå!!

                # Mirrored version
                dist_exp_to_sim_m, _ = sim_tree.query(exp3d_mirror,distance_upper_bound=distance_bound)
                dist_sim_to_exp_m, _ = exp_tree_mirror.query(sim_points,distance_upper_bound=distance_bound)

                n_unmatched_exp_m = np.sum(np.isinf(dist_exp_to_sim_m))
                n_unmatched_sim_m = np.sum(np.isinf(dist_sim_to_exp_m))
                matched_score_m = np.sum(dist_exp_to_sim_m[np.isfinite(dist_exp_to_sim_m)])
                # normalise score
                score_mirror = (matched_score_m + unmatched_penalty * (n_unmatched_exp_m + n_unmatched_sim_m))/n_total

                # Check score and keep only best score for each sim_frame
                if score < best_score:
                    best_score = score
                    ang = rot_idx * step_size_rad 
                    best_rotation = wrap_degrees(ang) 
                    mirror = 1.0

                #NOTE: OG her !
                if score_mirror < best_score:
                    best_score = score_mirror
                    ang = rot_idx * step_size_rad 
                    best_rotation = wrap_degrees(ang) 
                    mirror = -1.0
            # Store results for each sim_frame
            # nx4-shape [frame, score, rotation, mirror-factor]
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

    """
    step_size_rad = np.deg2rad(step_size)

    precomputed_rotated_vectors = [
        filter_sim(sim_frame, step_size_rad, reciprocal_radius)
        for sim_frame in simulated
    ]

    n_array = []

    for exp_vec in tqdm(experimental):
        exp3d = vector_to_3D(exp_vec, reciprocal_radius)
        exp3d_mirror = exp3d * np.array([-1,1,1])
        results = []

        for sim_idx, sim_rotations in enumerate(precomputed_rotated_vectors):
            best_score, best_rotation, mirror = float('inf'), 0.0, 1.0

            for rot_idx, sim3d in enumerate(sim_rotations):
                score = angular_score(exp3d, sim3d, angle_thresh_rad)
                score_mirror = angular_score(exp3d_mirror, sim3d, angle_thresh_rad)

                if score < best_score:
                    best_score = score
                    best_rotation = wrap_degrees(rot_idx * step_size_rad)
                    mirror = 1.0

                if score_mirror < best_score:
                    best_score = score_mirror
                    best_rotation = wrap_degrees(rot_idx * step_size_rad)
                    mirror = -1.0

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
    n_best: int
) -> np.ndarray:
    """
    This sucks
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
        exp3d = np.array(vector_to_3D(exp_vec,reciprocal_radius))
        # Mirror exp3d over the YZ-plane
        exp3d_mirror = exp3d * np.array([-1,1,1])
        results = []
        # Loop through each simulated frame
        for sim_idx, trees in enumerate(precomputed_trees):
            # just reset and declare these here to stop the lsp from bitching
            best_score, best_rotation, mirror = float('inf'), 0, 0.
            # Loop through each rotation of the simulated frame
            for rot_idx, tree in enumerate(trees):
                distances, _ = tree.query(exp3d)
                # low score is good
                score = np.sum(distances)
                # mirror score
                distances_mirror, _ = tree.query(exp3d_mirror)
                score_mirror = np.sum(distances_mirror)

                # Check score and keep only best score for each sim_frame
                if score < best_score:
                    best_score = score
                    ang = rot_idx * step_size_rad
                    best_rotation = wrap_degrees(ang)
                    mirror = 1.0
                # Check mirror score
                if score_mirror < best_score:
                    best_score = score
                    ang = rot_idx * step_size_rad
                    best_rotation = wrap_degrees(ang)
                    mirror = -1.0
            # Store results for each sim_frame
            # nx4-shape [frame, score, rotation, mirror-factor]
            results.append((sim_idx, best_score, best_rotation, mirror))
        # Sort by ascending score and select n_best
        results = sorted(results, key = lambda x : x[1])[:n_best]
        n_array.append(np.array(results))
    # Return array of shape (len(experimental), n_best, 4)
    return np.array(n_array)
 
