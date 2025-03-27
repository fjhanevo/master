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


#NOTE: Fjern, brukes ikke og er unødvendig
def apply_x_rotation(vector:np.ndarray,theta:float) -> np.ndarray:
    """
    It just rotates the sphere around the x-axis with a given angle
    """
    rot = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    return rot @ vector

#NOTE: Fjern, brukes ikke og er unødvendig
def apply_y_rotation(vector:np.ndarray, theta:float) -> np.ndarray:
    """
    It just rotates the sphere around the y-axis with a given angle
    """
    rot = np.array([[np.cos(theta), 0, np.sin(theta)], 
                    [0 ,1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    return rot @ vector

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


#NOTE: Litt usikker på hva denne gjør egt. blir mest sannsynlig fjerna
def save_rotate_sphere(vector:np.ndarray,step_size:float, filename:str) ->None:
    loop_list = np.arange(0,2*np.pi,step_size).tolist()
    save_vec = [] 
    for ang in loop_list:
        temp_vec = np.array([apply_z_rotation(vec,ang) for vec in vector])
        save_vec.append(temp_vec)
    save_vec=np.array(save_vec)
    # Expected output (M,N,3)
    print(save_vec.shape)
    np.save(file=filename,arr=save_vec, allow_pickle=True)

def kdtree_match(experimental, simulated, ang_step):
    best_match = None
    best_score = float('inf')
    rotation = None

    for exp_idx, exp_frame in enumerate(experimental):
        for sim_idx, sim_frame in enumerate(simulated):

            # KD-tree for fast nn search
            tree = cKDTree(sim_frame)
            
            # find nn in sim_frame for each point in exp_frame
            distances, _ = tree.query(exp_frame)

            # low score is good 
            score = np.sum(distances)

            # tracks best match
            if score < best_score:
                best_score = score
                # Endrer denne for nå fordi jeg bare vil ha sim_idx
                # best_match = (exp_idx,sim_idx)
                best_match = sim_idx
                rotation = exp_idx * ang_step

    return best_match, best_score, rotation

def kdtree_sim_rot(experimental, simulated, ang_step):
    best_match = 0
    best_score = float('inf')
    rotation = 0

    # Loop through stuff
    # for exp_idx, exp_frame in enumerate(experimental):
    for sim_idx, sim_frame in enumerate(simulated):
        for rot_idx, sim_rot in enumerate(sim_frame):
            # Build tree for every rotation
            tree = cKDTree(sim_rot)

            # find nn for exp
            distances, _ = tree.query(experimental)

            score = np.sum(distances)

            if score < best_score:
                best_score = score
                best_match = sim_idx
                rotation = rot_idx * ang_step

    return best_match, best_score, rotation

#TODO: n_best!!!
def kdtree_n_best(experimental, simulated, ang_step, n_best):
    """ 
    Matche algoritmen.
    SKAL FIKSE n_best ETTERHVERT!!!
    """

    # variables to store stuff
    num_sim_frames = len(simulated)
    frames = np.zeros(num_sim_frames, dtype=int)
    scores = np.full(num_sim_frames, float('inf'))
    rotations = np.zeros(num_sim_frames)
    best_frame = 0
    best_rotation = 0

    # loop twice to get to correct sim dim
    # inefficient but necessary cause ram is bad
    for sim_idx, sim_frame in enumerate(simulated):
        # now loop through the different rotations
        # reset the score after each sim_frame so it tracks the score per frame
        best_score = float('inf')
        
        for rot_idx, sim_rot in enumerate(sim_frame):
            # grow a tree for every rotation
            tree = cKDTree(sim_rot)

            # find nn for exp
            distances, _ = tree.query(experimental)

            # get score, low score is good
            score = np.sum(distances)

            # compare and update score
            if score < best_score:
                best_score = score
                best_rotation = rot_idx * ang_step
                best_frame = sim_idx
        # Save frame, score and rotation after each iteration
        frames[sim_idx] = best_frame
        scores[sim_idx] = best_score
        rotations[sim_idx] = best_rotation

        # Sort after score, and return n_best
        









def vector_match(experimental, simulated, ang_step, reciprocal_radius, n_best):
    """ 
    Takes in an experimental and simulated array, matches them and returns an 
    (len(experimental), n_best, nx4) array, which can be made into pyxem 
    OrientationMap object. 
    n_best KOMMER SNART!!!
    """
    result_lst = []

    # Assume sim is already 3d and rotated
    for exp_vec in experimental:
        # Transpose exp to 3D
        exp3d = np.array(vector_to_3D(exp_vec,reciprocal_radius))

        frame, score, rotation = kdtree_sim_rot(exp3d, simulated, ang_step)
        # frame, score, rotation = kdtree_sim_rot(exp3d, simulated, ang_step)

        lst = np.column_stack((frame, score, rotation, np.ones((1,))))
        result_lst.append(lst)
    n_array = np.array(result_lst)
    print(n_array.shape)
    return n_array


