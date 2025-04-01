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
    # First apply a full rotation to the experimental frame
    # exp = sm.full_z_rotation(exp,ang_step)
    # frames, score, rotation = sm.kdtree_match(exp,sim,ang_step)
    # frames, score, rotation = sm.kdtree_sim_rot(exp,sim,ang_step)
    # frames, score, rotation = sm.fast_kdtree_sim_rot(exp,sim,ang_step)
    frames, score, rotation = sm.vector_match(exp,sim,ang_step,reciprocal_radius, n_best)
    t2 = time()
    print(f"Computation time: {(t2-t1)/60} min")
    print('Best frames:', frames)
    print('Best score:', score)
    print('Rotation:', rotation)


def create_and_save_dataset(experimental, simulated, ang_step, reciprocal_radius, n_best, filename:str):
    n_array = sm.vector_match(experimental, simulated, ang_step, reciprocal_radius, n_best)
    np.save(file=filename, arr=n_array, allow_pickle=True)


if __name__ == '__main__':
    DIR_NPY = 'npy_files/'
    FILE_EXP = 'LF_peaks_m_center_m_peaks.npy'
    FILE_STRICT = 'LF_strict_peaks_log.npy'
    FILE_SIM = 'LF_r_theta_sim.npy'
    # FILE_SIM = 'filtered_simulation.npy'
    # FILE_SIM_ROT = 'sim_ang0005.npy'
    FILE_SIM_ROT = 'sim_rot_ang001.npy'
    FILE_SIM_ROT2 = 'sim_ang0005.npy'
    IN_PLANE_FILE = 'test_vector_match_ang0005.npy'

    experimental = np.load(DIR_NPY+FILE_EXP,allow_pickle=True)
    strict_peaks = np.load(DIR_NPY+FILE_STRICT, allow_pickle=True)
    # Quick polar transform
    experimental = sm.fast_polar(experimental)
    strict_peaks = sm.fast_polar(strict_peaks)
    simulated = np.load(DIR_NPY+FILE_SIM,allow_pickle=True)
    rot_simulated = np.load(DIR_NPY+FILE_SIM_ROT, allow_pickle=True)
    
    reciprocal_radius = 1.35 # [Å^-1]
    ang_step = 0.01
    # ang_step = 1    # for quick testing
    exp_frame =56 

    ## n_best = 1 dataset creation ##
    # n_best = 1 
    # t1 = time()
    # n1_array = sm.vector_match(experimental, rot_simulated, ang_step, reciprocal_radius, n_best)
    # print(n1_array.shape)
    # np.save(file=DIR_NPY+'ormap_n_best1_ang_step001.npy', arr=n1_array, allow_pickle=True)
    # t2 = time()
    # print(f"Computation time {(t2-t1)/60} min")
    # print(f"Computation time n_best = 1 {(t2-t1)/60} min")

    """ SETT PÅ DETTE NÅR DU KOMMER HJEM"""
    ## n_best = len(simulated) dataset creation ##
    n_best = len(simulated)
    t1 = time()
    n2_array = sm.vector_match(strict_peaks, rot_simulated, ang_step, reciprocal_radius, n_best)
    # n2_array = sm.vector_match(experimental, rot_simulated, ang_step, reciprocal_radius, n_best)
    print(n2_array.shape)
    np.save(file=DIR_NPY+'ormap_strict_rad2deg_n_best_all_ang_step001.npy', arr=n2_array, allow_pickle=True)
    t2 = time()
    print(f"Computation time file1: {(t2-t1)/60} min")

    ## Dataset ang0005
    ang_step = 0.005
    rot_simulated = np.load(DIR_NPY+FILE_SIM_ROT2, allow_pickle=True)
    n_best = len(simulated)
    t1 = time()
    n2_array = sm.vector_match(strict_peaks, rot_simulated, ang_step, reciprocal_radius, n_best)
    print(n2_array.shape)
    np.save(file=DIR_NPY+'ormap_strict_rad2deg_n_best_all_ang_step0005.npy', arr=n2_array, allow_pickle=True)
    t2 = time()
    print(f"Computation time file 2: {(t2-t1)/60} min")
    """"""








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


