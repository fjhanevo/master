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



def match_one_frame(exp, sim, ang_step, reciprocal_radius, n_best):
    t1 = time()
    n_array = sm.vector_match_one_frame(exp,sim,ang_step,reciprocal_radius, n_best)
    print(n_array.shape)
    t2 = time()
    frame = n_array[0][0][0]
    score = n_array[0][0][1]
    rotation = n_array[0][0][2]
    mirror = n_array[0][0][3]
    print(f"Computation time: {(t2-t1)/60} min")
    print('Best frame:', frame)
    print('Best score:',score) 
    print('Best rotation:', rotation)
    print('Mirror factor', mirror)
    return frame, score, rotation


def create_and_save_dataset(experimental, simulated, ang_step, reciprocal_radius, n_best, filename:str):
    n_array = sm.vector_match(experimental, simulated, ang_step, reciprocal_radius, n_best)
    np.save(file=filename, arr=n_array, allow_pickle=True)

def exp_and_sim_sphere_plot(exp, sim, rot, reciprocal_radius,lbls:tuple):
    exp3d = sm.vector_to_3D(exp,reciprocal_radius)
    sim_filtered = sim[~np.all(sim==0,axis=1)]
    sim_filtered3d = sm.vector_to_3D(sim_filtered, reciprocal_radius)
    sim_filtered3d_rot = np.array([sm.apply_z_rotation(vec,rot) for vec in sim_filtered3d])

    plotting.plot_spheres_with_axis_lims(sim_filtered3d_rot, exp3d,lbls)

if __name__ == '__main__':
    DIR_NPY = 'npy_files/'
    FILE_EXP = 'LF_peaks_m_center_m_peaks.npy'
    FILE_STRICT = 'peaks_all_LoG.npy'
    FILE_SIM = 'LF_r_theta_sim.npy'
    # FILE_SIM = 'filtered_simulation.npy'
    # FILE_SIM_ROT = 'sim_ang0005.npy'
    # FILE_SIM_ROT = 'sim_rot_ang001.npy'
    # FILE_SIM_ROT2 = 'sim_ang0005.npy'
    IN_PLANE_FILE = 'test_vector_match_ang0005.npy'

    experimental = np.load(DIR_NPY+FILE_STRICT,allow_pickle=True)
    # Quick polar transform
    experimental = sm.fast_polar(experimental)
    simulated = np.load(DIR_NPY+FILE_SIM,allow_pickle=True)
    
    reciprocal_radius = 1.35 # [Å^-1]
    ang_step = 1    # Degrees
    exp_frame =56 


    # filename = 'test_faster_method_with_0_5degs.npy'
    # create_and_save_dataset(experimental, simulated, ang_step, reciprocal_radius, len(simulated), DIR_NPY+filename)
    # t1 = time()
    # n_best = len()
    # ang_step = 0.5    # Now in degrees!!
    # filename = DIR_NPY + 'ormap_strict_ang0.5deg.npy'
    # create_and_save_dataset(experimental, simulated,ang_step, reciprocal_radius,n_best, filename)
    # t2 = time()
    # print(f"Computation time: {(t2-t1)/60} min")

    sim_frame, _, rotation = match_one_frame(experimental[exp_frame], simulated,np.deg2rad(ang_step), reciprocal_radius, len(simulated))
    sim_str = 'sim['+str(sim_frame)+']'
    exp_str = 'exp['+str(exp_frame)+']'
    ##### KEEP FOR LATER
    lbls = ('sim_str', 'exp_str')
    exp_and_sim_sphere_plot(experimental[exp_frame],simulated[int(sim_frame)],np.deg2rad(rotation),reciprocal_radius,lbls)


