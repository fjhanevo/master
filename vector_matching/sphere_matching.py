import pyxem as pxm
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from time import time

"""
Fil for sphere matching!:)
"""

def fast_polar(cart_vec):
    s = pxm.signals.DiffractionVectors(cart_vec)
    s = s.to_polar()
    return s.data

def vector_to_3D(vector:np.ndarray,reciprocal_radius:float) -> np.ndarray:
    """
    Takes in a 2D polar vector and converts it to 
    a 3D sphere.
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

    # create new dataset
    vector3d = np.stack([x,y,z],axis=-1)

    return vector3d

def apply_z_rotation(vector:np.ndarray,theta:float) -> np.ndarray:
    """
    It just rotates the sphere around the z-axis with a given angle
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot = np.array([[cos_theta, -sin_theta, 0],
                    [sin_theta, cos_theta, 0],
                    [0,0,1]])
    return vector @ rot.T

def full_z_rotation(vector:np.ndarray, ang_step:float) -> np.ndarray:
    angles = np.arange(0,2*np.pi,ang_step).tolist()
    return [apply_z_rotation(vector, theta) for theta in angles]

def filter_sim(sim:np.ndarray, ang_step:float, reciprocal_radius:float) -> np.ndarray:
    """
    Helper function for vector_match() to filter out zeros
    from sim because its homo, and now its in-homo
    """
    sim_filtered = sim[~np.all(sim == 0, axis=1)]

    sim_filtered_3d = vector_to_3D(sim_filtered, reciprocal_radius)

    return full_z_rotation(sim_filtered_3d,ang_step)

def vector_match_one_frame(experimental, simulated, ang_step, reciprocal_radius, n_best):
    """
    This is just for matching one frame and debugging
    """
    result_lst = []

    # Convert input degrees to radians
    ang_step = np.deg2rad(ang_step)
    # precompute KD-trees for all rotated sims
    t1 = time()
    precomputed_trees = [
        [cKDTree(rot_frame) for rot_frame in filter_sim(sim_frame, ang_step, reciprocal_radius)]
        for sim_frame in simulated
    ]
    t2 = time()
    print(f"Pre-compute time: {(t2-t1)/60} min")
    # Loop through all experimental vectors
    # Transpose to 3D
    exp3d = np.array(vector_to_3D(experimental,reciprocal_radius))
    # Mirror exp3d over YZ-plane
    exp3d_mirror = exp3d * np.array([-1,1,1])
    results = []

    # Loop through each simulated frame
    for sim_idx, trees in enumerate(precomputed_trees):
        # just reset and declare these here to stop the lsp from bitching
        best_score, best_rotation, mirror = float('inf'), 0, 1.0

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
                best_rotation = np.rad2deg(rot_idx * ang_step)
                mirror = 1.
                # in_plane = np.rad2deg(rot_idx * ang_step)
            if score_mirror < best_score:
                best_score = score_mirror
                best_rotation = np.rad2deg(rot_idx * ang_step)
                mirror = -1.

        # Store results for each sim_frame
        # nx4-shape [frame, score, rotation, mirror-factor]
        results.append((sim_idx, best_score, best_rotation, mirror))
    
    # Sort by ascending score and select n_best
    results = sorted(results, key = lambda x : x[1])[:n_best]
    result_lst.append(np.array(results))
    # Return array of shape (len(experimental), n_best, 4)
    n_array = np.array(result_lst)
    return n_array


def vector_match(experimental, simulated, ang_step, reciprocal_radius, n_best):
    """
    This is just for matching one frame and debugging
    """
    result_lst = []

    # Convert input degrees to radians
    ang_step = np.deg2rad(ang_step)
    # precompute KD-trees for all rotated sims
    t1 = time()
    precomputed_trees = [
        [cKDTree(rot_frame) for rot_frame in filter_sim(sim_frame, ang_step, reciprocal_radius)]
        for sim_frame in simulated
    ]
    t2 = time()
    print(type(precomputed_trees))
    print(f"Pre-compute time: {(t2-t1)/60}")
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
                    # Convert from rads to degs
                    best_rotation= np.rad2deg(rot_idx * ang_step)
                    mirror = 1.0

                # Check mirror score
                if score_mirror < best_score:
                    best_score = score_mirror
                    best_rotation= np.rad2deg(rot_idx * ang_step)
                    mirror = -1.0

            # Store results for each sim_frame
            # nx4-shape [frame, score, rotation, mirror-factor]
            results.append((sim_idx, best_score, best_rotation, mirror))
        
        # Sort by ascending score and select n_best
        results = sorted(results, key = lambda x : x[1])[:n_best]
        result_lst.append(np.array(results))
    # Return array of shape (len(experimental), n_best, 4)
    n_array = np.array(result_lst)
    return n_array


