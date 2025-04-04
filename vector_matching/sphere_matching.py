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
    # return rot @ vector
    return vector @ rot.T
#NOTE: Fjerner denne hvis den nye fungerer!
# def full_z_rotation(vector, ang_step):
#     """
#     This rotates the input vector 2pi with a given angular step size.
#     Makes a boring unrotated dataset of shape (N,3) to a cool new data set with 
#     shape (M,N,3) where the M is each new rotation.
#     """
#     loop_list = np.arange(0,360,ang_step).tolist() 
#     ret_vec = []
#     for ang in (loop_list):
#         temp_vec = np.array([apply_z_rotation(vec,ang) for vec in vector])
#         ret_vec.append(temp_vec)
#     return np.array(ret_vec)
#
# def full_z_rotation(vector, ang_step):
#     """
#     This rotates the input vector 2pi with a given angular step size.
#     Makes a boring unrotated dataset of shape (N,3) to a cool new data set with 
#     shape (M,N,3) where the M is each new rotation.
#     """
#
#     num_steps = int(360/ang_step)
#     ret_vec = np.empty((num_steps,len(vector), 3))
#     for i in range(num_steps):
#         ang = i * ang_step
#         temp_vec = np.array([apply_z_rotation(vec,ang) for vec in vector])
#         ret_vec[i] = temp_vec
#     return np.array(ret_vec)

#NOTE: Tester denne!
def full_z_rotation(vector:np.ndarray, ang_step:float) -> np.ndarray:
    angles = np.arange(0,2*np.pi,ang_step).tolist()
    return [apply_z_rotation(vector, theta) for theta in angles]

def filter_sim(sim:np.ndarray, ang_step:float, reciprocal_radius:float) -> np.ndarray:
    """
    Helper function for vector_match() to filter out zeros
    from sim because its homo, and now its in-homo
    """
    sim_filtered = sim[~np.all(sim == 0, axis=1)]

    # make it 3D
    sim_filtered_3d = vector_to_3D(sim_filtered, reciprocal_radius)

    # fully rotate sim3d
    # sim_filtered_3d_rot = full_z_rotation(sim_filtered_3d, ang_step)
    # return sim_filtered_3d_rot
    # print(sim_filtered_3d_rot.shape)
    return full_z_rotation(sim_filtered_3d,ang_step)


# def vector_match(experimental, simulated, ang_step, reciprocal_radius, n_best):
#     """
#     DENNE FUNKER BRA!
#
#     Params:
#         experimental: 
#             type: np.ndarray
#             (N,2) ndarray of polar exerimental dp's
#         simulated: 
#             type: np.ndarray
#             (M,N,2) ndarray of polar simulated dp's
#         ang_step:
#             type: float
#             angular step values from 0 to 2*pi
#         reciprocal_radius:
#             type: float
#             reciprocal radius to take into account
#         n_best:
#             type: int
#             amount of frames to keep score of
#     Returns:
#         n_array:
#         type: np.ndarray
#         Return an nx4 array of shape (len(experimental), n_best, 4)
#         where the nx4 array is [index, score, in-plane rotation, mirror-factor]
#
#     """
#     result_lst = []
#
#     # Loop through all experimental vectors
#     for exp_vec in tqdm(experimental):
#         print(exp_vec.shape)
#         # Transpose to 3D
#         exp3d = np.array(vector_to_3D(exp_vec,reciprocal_radius))
#         results = []
#
#         # Loop through each simulated frame
#         for sim_idx, sim_frame in enumerate(simulated):
#             # filter out zeros and make sim 3d and rotated
#             sim_processed = filter_sim(sim_frame, ang_step, reciprocal_radius)
#             # just reset and declare these here to stop the lsp from bitching
#             best_score = float('inf')
#             in_plane = 0
#
#             # Loop through each rotation of the simulated frame
#             for rot_idx, rot_frame in enumerate(sim_processed):
#                 # Build tree for nn-search
#                 tree = cKDTree(rot_frame)
#                 
#                 distances, _ = tree.query(exp3d)
#                 
#                 # low score is good
#                 score = np.sum(distances)
#
#                 # Check score and keep only best score for each sim_frame
#                 if score < best_score:
#                     best_score = score
#                     in_plane = rot_idx * ang_step
#                     # Convert from rads to degs
#                     # in_plane = np.rad2deg(rot_idx * ang_step)
#
#             # Store results for each sim_frame
#             # nx4-shape [frame, score, rotation, mirror-factor]
#             results.append((sim_idx, best_score, in_plane, 1.))
#         
#         # Sort by ascending score and select n_best
#         results = sorted(results, key = lambda x : x[1])[:n_best]
#
#         result_lst.append(np.array(results))
#
#     # Return array of shape (len(experimental), n_best, 4)
#     n_array = np.array(result_lst)
#     return n_array
#


def vector_match_one_frame(experimental, simulated, ang_step, reciprocal_radius, n_best):
    """
    This is just for matching one frame and debugging
    """
    result_lst = []

    # precompute KD-trees for all rotated sims
    precomputed_trees = [
        [cKDTree(rot_frame) for rot_frame in filter_sim(sim_frame, ang_step, reciprocal_radius)]
        for sim_frame in simulated
    ]
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


#NOTE: Denne baserer seg på at jeg gjort endringer i apply_z_rotation etc.!
# Beholder den forrige foreløpig.
def vector_match(experimental, simulated, ang_step, reciprocal_radius, n_best):
    """
    This is just for matching one frame and debugging
    """
    result_lst = []

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
    # Transpose to 3D
    for exp_vec in tqdm(experimental):
        exp3d = np.array(vector_to_3D(exp_vec,reciprocal_radius))
        # print(exp3d.shape)
        results = []

        # Loop through each simulated frame
        for sim_idx, trees in enumerate(precomputed_trees):
            # just reset and declare these here to stop the lsp from bitching
            best_score, best_rotation = float('inf'), 0

            # Loop through each rotation of the simulated frame
            for rot_idx, tree in enumerate(trees):
                distances, _ = tree.query(exp3d)

                
                # low score is good
                score = np.sum(distances)

                # Check score and keep only best score for each sim_frame
                if score < best_score:
                    best_score = score
                    best_rotation = rot_idx * ang_step
                    # Convert from rads to degs
                    # in_plane = np.rad2deg(rot_idx * ang_step)

            # Store results for each sim_frame
            # nx4-shape [frame, score, rotation, mirror-factor]
            results.append((sim_idx, best_score, best_rotation, 1.))
        
        # Sort by ascending score and select n_best
        results = sorted(results, key = lambda x : x[1])[:n_best]
        result_lst.append(np.array(results))
    # Return array of shape (len(experimental), n_best, 4)
    n_array = np.array(result_lst)
    return n_array


