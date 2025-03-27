import pyxem as pxm
import numpy as np
import plotting
import sphere_matching as sm
from time import time
import matplotlib.pyplot as plt



"""
Fil for å teste ut ting:)
"""

def make_rotating_sphere_gif(vec1:np.ndarray,vec2:np.ndarray,labels:tuple) -> None:
    """
    Takes in two 3D vectors and rotates one vector
    around the z-axis 
    """ 

    l1, l2 = labels
    t1 = time()
    loop_list = np.arange(0,2*np.pi,0.1).tolist()
    for ang_step in loop_list:
        # Rotate sphere 2
        filename = 'ang_' + str(ang_step) + '.png'
        labels = (l1, l2, filename)
        ##### Use this to plot two spheres #######
        sphere_z = np.array([sm.apply_z_rotation(v2,ang_step) for v2 in vec2])
        plotting.plot_spheres_to_gif(vec1,sphere_z,labels)

        ##### This is for fun hihihihihi ######

        # sphere_x = np.array([apply_x_rotation(v2,ang_step) for v2 in vec2])
        # sphere_xy = np.array([apply_y_rotation(v2,ang_step) for v2 in sphere_x])
        # sphere_xyz = np.array([apply_z_rotation(v2,ang_step) for v2 in sphere_xy])
        # plotting.plot_spheres_to_gif(vec1,sphere_xyz,labels)
    t2 = time()

    print(f"Computation time: {(t2-t1)/60} min")



def run_full_tree(experimental, simulated, reciprocal_radius, ang_step):
    in_plane = []
    for idx in range(experimental.shape[0]):
        # convert 2D to 3D
        frame = np.array(sm.vector_to_3D(experimental[idx], reciprocal_radius))
        # fully rotate the frame
        rot_frame = sm.full_z_rotation(frame, ang_step)
        # debuggggggggggg
        print(rot_frame.shape)

        # now we can use kdtrees to match
        ## WILL only track rotation this time!
        _,_, rot = sm.kdtree_match(rot_frame, simulated, ang_step)
        
        # save rot at each iteration
        in_plane.append(rot)

    # assuming this takes a long time to run so will save in_plane
    in_plane = np.array(in_plane)
    np.save(file='full_in_plane_test.npy',arr=in_plane,allow_pickle=True)
    print("We did it!")

def match_one_frame(exp, sim, ang_step):
    t1 = time()
    # First apply a full rotation to the experimental frame
    # exp = sm.full_z_rotation(exp,ang_step)
    # frames, score, rotation = sm.kdtree_match(exp,sim,ang_step)
    # frames, score, rotation = sm.kdtree_sim_rot(exp,sim,ang_step)
    # frames, score, rotation = sm.fast_kdtree_sim_rot(exp,sim,ang_step)
    frames, score, rotation = sm.slightly_fast_kdtree(exp,sim,ang_step)
    t2 = time()
    print(f"Computation time: {(t2-t1)/60} min")
    print('Best frames:', frames)
    print('Best score:', score)
    print('Rotation:', rotation)

def create_result_dataset(exp, sim, ang_step, reciprocal_radius):
    result_lst = []

    # First turn exp and sim to 3D
    sim3d = np.array([sm.vector_to_3D(sim_vec,reciprocal_radius) for sim_vec in sim])
    # We do it like this cause its inhomogeneous
    for exp_vec in exp:
        exp3d = np.array(sm.vector_to_3D(exp_vec, reciprocal_radius))
        # Apply a full z-rotation to each frame
        exp3d = sm.full_z_rotation(exp3d,ang_step)
        frame, score, rotation = sm.kdtree_match(exp3d,sim3d,ang_step)
        # Create nx4 array ['index', 'correlation', 'in-plane', mirror_factor]
        lst = np.array([[frame, score, rotation, 1.0]])
        # Append the data 
        result_lst.append(lst)
    n_array = np.array(result_lst)
    print(n_array.shape)
    return n_array

def full_create_dataset(exp, sim, ang_step, reciprocal_radius):
    results = []

    # Convert sim to 3D
    sim3d = np.array([sm.vector_to_3D(sim_vec, reciprocal_radius) for sim_vec in sim])

    for exp_vec in exp:
        # Convert to 3D like this cause its inhomogeneous
        exp3d = np.array(sm.vector_to_3D(exp_vec, reciprocal_radius))
        # Apply a full z-rotation
        exp3d = sm.full_z_rotation(exp3d, ang_step)

        # Track best matches for all simulated frames
        frames, scores, rotations = sm.full_kdtree_match(exp3d, sim3d,ang_step)

        # Create nx4 array ['index', 'correlation', 'in-plane', mirror_factor]
        lst = np.column_stack((frames, scores, rotations, np.ones_like(frames)))
        results.append(lst)
    n_array = np.array(results)
    print(n_array.shape)
    return n_array

def alt_create_dataset(exp, sim, ang_step, reciprocal_radius):
    results = []

    # Rotate sim3d 
    sim3d = np.array([sm.vector_to_3D(sim_vec,reciprocal_radius) for sim_vec in sim])

    for exp_vec in exp:
        exp3d = np.array(sm.vector_to_3D(exp_vec, reciprocal_radius))
        # Apply a full z-rotation to each frame
        exp3d = sm.full_z_rotation(exp3d,ang_step)
        frame, score, rotation = sm.kdtree_match(exp3d,sim3d,ang_step)
        lst = np.column_stack((frame, score, rotation, np.ones_like(frame)))
        results. append(lst)
    n_array = np.array(results)
    print(n_array.shape)
    return n_array



if __name__ == '__main__':
    DIR_NPY = 'npy_files/'
    FILE_EXP = 'LF_peaks_m_center_m_peaks.npy'
    FILE_STRICT = 'LF_strict_peaks_log.npy'
    FILE_SIM = 'LF_r_theta_sim.npy'
    # IN_PLANE_FILE = 'full_in_plane_test.npy'
    # IN_PLANE_FILE = 'ang_step0_01_factor1_negative_rot.npy'
    IN_PLANE_FILE = 'test_vector_match_ang0005.npy'

    
    experimental = np.load(DIR_NPY+FILE_EXP,allow_pickle=True)
    # strict_peaks = np.load(DIR_NPY+FILE_STRICT, allow_pickle=True)
    # Quick polar transform
    experimental = sm.fast_polar(experimental)
    # strict_peaks = sm.fast_polar(strict_peaks)
    simulated = np.load(DIR_NPY+FILE_SIM,allow_pickle=True)
    
    reciprocal_radius = 1.35 # [Å^-1]
    ang_step = 0.01
    exp_frame =56 

    ## Check frame output
    sim3d = np.array([sm.vector_to_3D(sim,reciprocal_radius) for sim in simulated])
    # exp3d = np.array(sm.vector_to_3D(experimental[exp_frame],reciprocal_radius))
    # match_one_frame(exp3d,sim3d, ang_step)
    # print("----------")
    t1 = time()
    ang_step = 0.005
    rot_sim_0005 = np.array([sm.full_z_rotation(sim, ang_step) for sim in sim3d])
    np.save(file=DIR_NPY+'sim_rot_ang0005.npy', arr=rot_sim_0005, allow_pickle=True)
    print(rot_sim_0005.shape)
    t2 = time()
    print(f"Computation time {(t2-t1)/60} min")








    ## Make dataset
    # t1 = time()
    # results = sm.vector_match(experimental, simulated, ang_step, reciprocal_radius)
    # t2 = time()
    # print(f"Computation time {(t2-t1)/60} min")
    # np.save(file=DIR_NPY+IN_PLANE_FILE, arr=results,allow_pickle=True)
    # print("File saved:",IN_PLANE_FILE)
   


    ##### KEEP FOR LATER
    # sim3d = np.array([sm.vector_to_3D(sim,reciprocal_radius) for sim in simulated])
    # exp3d = np.array(sm.vector_to_3D(experimental[exp_frame],reciprocal_radius))
    # match_one_frame(exp3d,sim3d, ang_step)
    # lbls = ('sim[263]','exp[56]')
    # lbls = ('sim522', 'exp56')
    # exp3d_rot= np.array([sm.apply_z_rotation(v,4.92) for v in exp3d])
    # plotting.plot_spheres_with_axis_lims(sim3d[522],exp3d_rot, lbls)
    # sim3d_rot= np.array([sm.apply_z_rotation(v,-1.8) for v in sim3d[263]])
    # plotting.plot_spheres_with_axis_lims(sim3d_rot,exp3d, lbls)
    # make_rotating_sphere_gif(sim3d[263],exp3d,lbls)


