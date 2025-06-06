import gc
import numpy as np
import plotting
import sphere_matching as sm
from time import time
import vector_match as vm

"""
Fil for å teste ut ting:)
"""

def make_rotating_sphere_gif(vec1:np.ndarray,vec2:np.ndarray, reciprocal_radius:float,labels:tuple) -> None:
    """
    Takes in two 2D vectors and rotates vec1
    around the z-axis 
    """ 
    # Remove 0's
    vec1= vec1[~np.all(vec1==0,axis=1)]
    # To 3D
    vec13d= sm.vector_to_3D(vec1, reciprocal_radius)
    vec23d = sm.vector_to_3D(vec2,reciprocal_radius)

    l1, l2 = labels
    t1 = time()
    loop_list = np.arange(0,2*np.pi,0.1).tolist()
    for ang_step in loop_list:
        # Rotate sphere 1
        filename = 'ang_' + str(ang_step) + '.png'
        labels = (l1, l2, filename)
        ##### Use this to plot two spheres #######
        sphere_z = np.array([sm.apply_z_rotation(v1,ang_step) for v1 in vec13d])
        plotting.plot_spheres_to_gif(sphere_z,vec23d,labels)
    t2 = time()

    print(f"Computation time: {(t2-t1)/60} min")

def match_one_frame(exp, sim, ang_step, reciprocal_radius, n_best, method):
    t1 = time()
    n_array = sm.vector_match(exp,sim,ang_step,reciprocal_radius,n_best, method)
    print(n_array.shape)
    t2 = time()
    n_best = n_array[0][0]
    frame, score, rotation, mirror = n_best[0],n_best[1],n_best[2], n_best[3]
    print(f"Computation time: {(t2-t1)/60} min")
    print('Best frame:', frame)
    print('Best score:',score) 
    print('Best rotation:', rotation)
    print('Mirror factor:', mirror)
    return int(frame), score, rotation, mirror

def create_and_save_dataset(
    experimental: np.ndarray, 
    simulated: np.ndarray, 
    step_size: float, 
    reciprocal_radius: float, 
    n_best: int,
    method: str, 
    filename:str
) -> None:
    t1 = time()
    n_array = vm.vector_match(experimental, simulated, step_size, reciprocal_radius, n_best, method)
    print(n_array.shape)
    np.save(file=filename, arr=n_array, allow_pickle=True)
    t2 = time()
    print(f"Computation time {(t2-t1)/60} min")
    # Free memory
    del n_array
    gc.collect()

def exp_and_sim_sphere_plot(exp, sim, rot, reciprocal_radius,mirror,lbls:tuple):
    exp3d = sm.vector_to_3D(exp,reciprocal_radius)
    # Back to radians for consistency
    rot = np.deg2rad(rot)
    if mirror < 0.0:
        exp3d *= np.array([1,-1,1])
    sim_filtered = sim[~np.all(sim==0,axis=1)]
    sim_filtered3d = sm.vector_to_3D(sim_filtered, reciprocal_radius)
    sim_filtered3d_rot = np.array([sm.apply_z_rotation(vec,rot) for vec in sim_filtered3d])

    plotting.plot_spheres_with_axis_lims(sim_filtered3d_rot, exp3d,lbls)

def dataset_for_algorithm_accuracy(simulated, step_size, reciprocal_radius,n_best, filename):
    # Filter part of sim that is going to be matched
    sim_filtered = []
    for idx in range(simulated.shape[0]):
        r_theta = simulated[idx]

        # mask out zeroes
        mask = ~np.all(r_theta == 0, axis = 1)

        # apply mask
        r_theta_masked = r_theta[mask]

        sim_filtered.append(r_theta_masked)

    sim_filtered = np.array(sim_filtered, dtype=object)

    create_and_save_dataset(
        experimental=sim_filtered, 
        simulated=simulated, 
        step_size=step_size,
        reciprocal_radius=reciprocal_radius,
        n_best=n_best,
        method="",
        filename=filename,
    )

if __name__ == '__main__':
    DIR_NPY = 'npy_files/'
    FILE_STRICT = 'peaks_all_LoG.npy'
    FILE_INTENSITY = 'peaks_intensity_all_LoG.npy'
    FILE_SIM = 'LF_r_theta_sim.npy'
    FILE_SIM_INTENSITY = 'sim_r_theta_intensity.npy'
    FILE_ORMAP = 'ormap_step05deg_dist005_penalty075.npy'

    experimental = np.load(DIR_NPY+FILE_STRICT,allow_pickle=True)
    # Quick polar transform
    # experimental = sm.fast_polar(experimental)
    simulated = np.load(DIR_NPY+FILE_SIM,allow_pickle=True)
    
    reciprocal_radius = 1.35 # [Å^-1]
    step_size = 0.5    # Degrees
    exp_frame = 56
    n_best = len(simulated) 

    
    exp_intensity = np.load(DIR_NPY+FILE_INTENSITY,allow_pickle=True)
    sim_intensity = np.load(DIR_NPY+FILE_SIM_INTENSITY,allow_pickle=True)

    exp_intensity = sm.fast_polar(exp_intensity)
    # match_one_frame([experimental[exp_frame]],simulated,step_size,reciprocal_radius,n_best,1)

    ### CREATE NEW FILES ### 
    filename = '030625_vector_match_score_intensity_step05deg_dist005.npy'
    # filename = 'temp.npy'
    create_and_save_dataset(
        exp_intensity,
        sim_intensity, 
        step_size,
        reciprocal_radius,
        n_best, 
        method="score_intensity",
        filename=DIR_NPY+filename,
    )

    # sim_frame = 1087
    # rotation = 288.5
    # mirror = -1.0
    # lbls  = ('simulated', 'experimental')
    # exp_and_sim_sphere_plot(experimental[exp_frame],simulated[sim_frame],0, reciprocal_radius, mirror, lbls )
    # exp_and_sim_sphere_plot(experimental[exp_frame],simulated[sim_frame],rotation, reciprocal_radius, mirror, lbls )
