import pyxem as pxm
import numpy as np
from scipy.spatial import cKDTree

"""
Fil for sphere matching!:)
"""


def vector_to_3D(vector:np.ndarray,reciprocal_radius:float) -> np.ndarray:
    """
    Takes in a 2D polar vector and converts it to 
    a 3D sphere.
    """
    R = reciprocal_radius
    # get r coords
    r = vector[...,0]
    # get theta coords
    theta = vector[...,1]

    l = 2*np.arctan(r/(2*R))

    # 3D coords
    x = np.sin(l)*np.cos(theta)
    y = np.sin(l)*np.sin(theta)
    z = np.cos(l)

    # create new dataset
    vector3d = np.stack([x,y,z],axis=-1)

    return vector3d

def fast_polar(cart_vec):
    s = pxm.signals.DiffractionVectors(cart_vec)
    s = s.to_polar()
    return s.data


def apply_z_rotation(vector:np.ndarray,theta:float) -> np.ndarray:
    """
    It just rotates the sphere around the z-axis with a given angle
    """
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0,0,1]])
    return rot @ vector

#TODO: Se om denne kan speedes opp, krever mye RAM og tar tid å kjøre den
def full_z_rotation(vector, ang_step):
    """
    This rotates the input vector 2pi with a given angular step size.
    Makes a boring unrotated dataset of shape (N,3) to a cool new data set with 
    shape (M,N,3) where the M is each new rotation.
    """
    loop_list = np.arange(0,2*np.pi,ang_step).tolist() 
    ret_vec = []
    for ang in loop_list:
        temp_vec = np.array([apply_z_rotation(vec,ang) for vec in vector])
        ret_vec.append(temp_vec)
    return np.array(ret_vec)

def vector_match(experimental, simulated, ang_step, reciprocal_radius, n_best):
    """
    DENNE FUNKER BRA!

    """
    result_lst = []

    # Loop through all experimental vectors
    for exp_vec in experimental:
        # Transpose to 3D
        exp3d = np.array(vector_to_3D(exp_vec,reciprocal_radius))
        results = []

        # Loop through each simulated frame
        for sim_idx, sim_frame in enumerate(simulated):
            # just reset and declare these here to stop the lsp from bitching
            # best_frame = 0
            best_score = float('inf')
            in_plane = 0

            # Loop through each rotation of the simulated frame
            for rot_idx, rot_frame in enumerate(sim_frame):
                # Build tree for nn-search
                tree = cKDTree(rot_frame)
                
                distances, _ = tree.query(exp3d)
                
                # low score is good
                score = np.sum(distances)

                # Check score and keep only best score for each sim_frame
                if score < best_score:
                    best_score = score
                    # Convert from rads to degs
                    in_plane = np.rad2deg(rot_idx * ang_step)
                    # in_plane = rot_idx * ang_step

            # Store results for each sim_frame
            # nx4-shape [frame, score, rotation, mirror-factor]
            results.append((sim_idx, best_score, in_plane, 1.))
        
        # Sort by ascending score and select n_best
        results = sorted(results, key = lambda x : x[1])[:n_best]

        result_lst.append(np.array(results))

    # Return array of shape (len(experimental), n_best, 4)
    n_array = np.array(result_lst)
    return n_array


