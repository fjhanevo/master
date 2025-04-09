import pyxem as pxm
import numpy as np
from scipy.spatial import cKDTree
from time import time
from plotting import plot_2D_plane, plot_2D_plane_save

"""
File for vector matching...
"""

def cart2pol(cart_vec:np.ndarray) -> np.ndarray:
    s = pxm.signals.DiffractionVectors(cart_vec)
    s = s.to_polar()
    return s.data

def rotation_matrix(vec:np.ndarray, theta:float) -> np.ndarray:
    """
    2D rotation matrix rotating vec by angle theta
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s,c)))
    return vec @ R.T

def full_rotation(vec:np.ndarray, step_size:float) -> np.ndarray:
    angles = np.arange(0, 2*np.pi, step_size).tolist()
    return [rotation_matrix(vec, theta) for theta in angles]

def filter_sim(vec:np.ndarray, step_size:float) -> np.ndarray:
    vec_filtered = vec[~np.all(vec == 0, axis=1)]
    return full_rotation(vec_filtered,step_size)

def vm_one_frame(exp, sim, step_size):
    result_lst = []
    
    step_size_rad = np.deg2rad(step_size)

    t1 = time()
    pre_com_trees = [
        [cKDTree(rot_frame) for rot_frame in filter_sim(sim_frame, step_size_rad)]
        for sim_frame in sim
    ]
    t2 = time()
    print(f"Pre-compute time: {(t2-t1)/60} min")


    exp_mirror = exp * np.array([-1,1])
    results = []

    for sim_idx, trees in enumerate(pre_com_trees):
        best_score, best_rotation, mirror = float('inf'), 0, 1.0
        for rot_idx, tree in enumerate(trees):
            distances, _ = tree.query(exp)
            distances_mirror, _ = tree.query(exp_mirror)

            score = np.sum(distances)
            score_mirror = np.sum(distances_mirror)
            
            if score < best_score:
                best_score = score
                best_rotation = rot_idx * step_size
                mirror = 1.0

            if score_mirror < best_score:
                best_score = score_mirror
                best_rotation = rot_idx * step_size
                mirror = -1.0
        results.append((sim_idx, best_score, best_rotation, mirror))
    results = sorted(results, key = lambda x : x[1])
    result_lst.append(np.array(results))
    n_array = np.array(result_lst)
    return n_array

def match_one_frame(exp, sim, step_size):
    t1 = time()
    n_array = vm_one_frame(exp, sim, step_size)
    print(n_array.shape)
    t2 = time()
    print(f"Computation time: {(t2-t1)/60} min")
    n = n_array[0][0]
    frame, score, rotation, mirror  = n[0], n[1], n[2], n[3]
    print('Best frame:', frame)
    print('Best score:', score)
    print('Best rotation:', rotation)
    print('Mirror:', mirror)
    return frame, rotation, mirror

def plot_rotation_gif(vec1, vec2, step_size,mirror,labels):
    l1, l2 = labels
    vec1 = vec1 * np.array([mirror,1])
    step_size = np.deg2rad(step_size)
    loop_lst = np.arange(0,2*np.pi,step_size).tolist()
    vec2= vec2[~np.all(vec2==0,axis=1)]
    for ang_step in loop_lst:
        rot = rotation_matrix(vec2,ang_step)
        filename = 'ang_'+str(ang_step)+'.png'
        lbls = (l1,l2,filename)
        plot_2D_plane_save(vec1,rot,lbls)




if __name__ == '__main__':
    DIR_NPY = 'npy_files/'
    FILE_SIM = 'LF_r_theta_sim.npy'
    FILE_EXP = 'peaks_all_LoG.npy'

    exp_data = np.load(DIR_NPY+FILE_EXP, allow_pickle=True)
    sim_data = np.load(DIR_NPY+FILE_SIM, allow_pickle=True)
    # exp to 2d polar, sim already polar
    exp_data = cart2pol(exp_data)

    exp_frame = 29 
    reciprocal_radius = 1.35
    step_size = 5 # degree

    lbls = ('exp','sim')
    # plot_2D_plane(exp_data[exp_frame], sim_data[4106],lbls)

    sim_frame, rotation, mirror =  match_one_frame(exp_data[exp_frame], sim_data, step_size)
    sim_rot = rotation_matrix(sim_data[int(sim_frame)],rotation)
    exp_lbl = 'exp['+str(int(exp_frame))+']'
    sim_lbl = 'sim['+str(int(sim_frame))+']'
    lbls = (exp_lbl, sim_lbl)
    plot_rotation_gif(exp_data[exp_frame], sim_data[int(sim_frame)],step_size,mirror,lbls)
