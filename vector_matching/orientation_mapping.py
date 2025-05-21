import hyperspy.api as hs
import pyxem as pxm
import numpy as np
import simulation as sim
import plotting as plot
import matplotlib.pyplot as plt
from sphere_matching import vector_match_kd, fast_polar
from time import time



def to_orientation_map(data, simulation):
    """
    Converts numpy array into OrientaionMap and adds metdata.
    """
    data = pxm.signals.indexation_results.OrientationMap(data)
    # Add metadata
    data.metadata.VectorMetadata.column_names = ['index', 'correlation', 'rotation', 'factor']
    data.metadata.VectorMetadata.units = ['a.u', 'a.u', 'deg', 'a.u']
    data.simulation = simulation

    return data

def compare_orientations(tm_orientation:np.ndarray, vm_orientation:np.ndarray) -> dict:
    """
    Commpares orientation from TM and VM
    Params: 
        tm_orientation: np.ndarray
            template matching results array (N,n_best,4)
        vm_orientation: np.ndarray
    Returns:
        dict
            summary of stats of the comparison
    """
    assert tm_orientation.shape == vm_orientation.shape


    # Extract rotation angles in degrees only includes the n_best
    tm_rot = tm_orientation[:,:,2]
    vm_rot = vm_orientation[:,:,2]

    # Compute rotation difference, accounting for wrap-around at 360deg
    diff = np.abs(tm_rot - vm_rot) % 360
    diff = np.where(diff > 180, 360 - diff, diff)   # take shortest angle

    # Optionally: check simulation indices too
    sim_match = tm_orientation[:,:, 0] == vm_orientation[:,:,0]

    # Stats
    stats = {
        "mean_abs_rotation_diff_deg": np.mean(diff),
        "median_abs_rotation_diff_deg": np.median(diff),
        "max_rotation_diff_deg": np.max(diff),
        "min_rotation_diff_deg": np.min(diff),
        "percent_sim_idx_match": 100*np.mean(sim_match)
    }

    plt.figure(figsize=(8,4))
    plt.hist(diff.flatten(), bins=30, edgecolor='black')
    plt.xlabel("Absolute Rotation Difference (degrees)")
    plt.ylabel("Frequency")
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return stats 

def check_overlay_plot(exp, frame, sim, simulation):
    # Match one frame
    t1 = time()
    sim_trimmed = sim[:31]   # did this to decrease comp time, as I know the answer
    print(sim_trimmed.shape)
    n_array = vector_match_kd([exp[frame]], sim_trimmed,step_size=0.5, reciprocal_radius=1.35, n_best=1)
    t2 = time()
    print(n_array.shape)
    n_best = n_array[0][0]
    frame_found, score, rotation, mirror = n_best[0],n_best[1],n_best[2], n_best[3]
    print(f"Computation time: {(t2-t1)/60} min")
    print('Best frame:', frame_found)
    print('Best score:',score) 
    print('Best rotation:', rotation)
    print('Mirror factor:', mirror)

    n_ormap = to_orientation_map(n_array, simulation)
    # load org dataset
    s = hs.load('processed_hspy_files/LeftFish_unmasked.hspy')
    # Take out the frame
    s_frame = s.inav[frame:frame+1]

    s_frame.plot(cmap='viridis_r', norm='log', title='', colorbar=False, scalebar_color='black', axes_ticks='off')
    s_frame.add_marker(n_ormap.to_markers(annotate=True))
    plt.show()



if __name__ == '__main__':
    DIR_NPY = 'npy_files/'
    DIR_HSPY = 'processed_hspy_files/'
    HSPY = 'LF_cal_log_m_center_strict_peaks.hspy'
    ORG_HSPY = 'LeftFish_unmasked.hspy'
    FILE_KD = '140525_vector_match_kd_step05deg_distbound005_fixedmirror.npy'
    FILE_ANG = '150525_vector_match_ang_score_step05deg_angtresh005_fixedmirror.npy'
    FILE_SUM = '180525_vector_match_sum_score_step05deg_fixedmirror.npy'

    hs.set_log_level('WARNING')
    s = hs.load(DIR_HSPY+HSPY)
    
    ### SIMULATED ###
    s_pol = s.get_azimuthal_integral2d(npt=112, radial_range=(0.,1.35))
    phase = sim.unit_cell()
    grid, orientation = sim.gen_orientation_grid(phase)
    simgen = sim.get_simulation_generator(precession_angle=1., minimum_intensity=1e-4, approximate_precession=True)
    simulation = sim.compute_simulations(simgen, phase, grid, reciprocal_radius=1.35,
                                      max_excitation_error=0.05)

    sim_results = s_pol.get_orientation(simulation,n_best=grid.size,frac_keep=1.)  # Creates an OrientationMap

    
    frame = 29


    i, j = frame, frame+1
    
    ### EXPERIMENTAL ###
    exp_results = np.load(DIR_NPY+FILE_KD, allow_pickle=True)
    exp_results = to_orientation_map(exp_results,simulation)
    # exp2 = np.load(DIR_NPY+'150525_vector_match_ang_score_step05deg_angtresh005_fixedmirror.npy', allow_pickle=True)
    # exp3 = np.load(DIR_NPY+'060525ormap_step05deg_vector_match_sum_score_NO_MIRROR_wrap_degrees_v060525.npy', allow_pickle=True)
    # exp2 = to_orientation_map(exp2, simulation)
    # exp3 = to_orientation_map(exp3, simulation)
    lbls = ('Score A', 'Score B', 'Score C')
    clrs = ('Blue', 'Green', 'Red')
    # print("exp:", exp_results.data[frame][0])
    # print("sim:", sim_results.data[frame][0])

    ### PLOTS ### 
    plot.plot_ipf_misorientations(exp_results, phase, cmap='viridis_r')
    # plot.plot_misorientation_scatter(exp_results)
    # plot.plot_misorientation_scatter(sim_results)
    # plot.plot_ipf(sim_results,frame,phase,orientation, 'viridis')
    # plot.plot_ipf(exp_results,frame,phase,orientation, 'viridis_r')
    # plot.plot_with_markers(exp_results,DIR_HSPY+ORG_HSPY,i,j)
    # plot.plot_with_markers(sim_results,DIR_HSPY+ORG_HSPY,i,j)
    
