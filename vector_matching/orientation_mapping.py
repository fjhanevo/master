import hyperspy.api as hs
import pyxem as pxm
import numpy as np
import simulation as sim
import plotting as plot
import matplotlib.pyplot as plt
import vector_match as vm
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

def check_overlay_plot(exp, frame, simulation, phase, orientation, method):
    # Match one frame
    sim_data = np.load(DIR_NPY+'LF_r_theta_sim.npy', allow_pickle=True)
    t1 = time()
    sim_trimmed = sim_data[:1300]   # did this to decrease comp time, as I know the answer
    # n_array = vector_match_kd([exp[frame]], sim_trimmed,step_size=0.5, reciprocal_radius=1.35, n_best=1)
    n_array = vm.vector_match(
        [exp[frame]], 
        sim_data,
        step_size=0.5, 
        reciprocal_radius=1.35, 
        n_best=len(sim_data), 
        method=method)
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

    # plot ipf
    plot.plot_ipf(n_ormap, 0, phase, orientation, cmap='viridis_r')


def get_misorientation_statistics(data):
    """
    Computes some cool misorientation statistics
    """

    loris = data.to_single_phase_orientations()
    loris_best = loris[:, 0]
    loris_ang = loris_best.angle_with_outer(loris_best, degrees=True)

    stats = {}
    # use only neighboring misorientations
    misorientations = np.array([loris_ang[i, i+1] for i in range(len(loris_ang)-1)])
    stats['mean'] = np.mean(misorientations)
    stats['min'] = np.min(misorientations)
    stats['max'] = np.max(misorientations)
    stats['std'] = np.std(misorientations)

    # thresholds
    t1, t2 = 5.0, 2.5
    below_t1deg = np.sum(misorientations < t1)
    stats['precent_below_5deg'] = 100.0 * below_t1deg/ len(misorientations)
    below_t2deg = np.sum(misorientations < t2)
    stats['precent_below_2.5deg'] = 100.0 * below_t2deg/ len(misorientations)

    return stats


if __name__ == '__main__':
    DIR_NPY = 'npy_files/'
    DIR_HSPY = 'processed_hspy_files/'
    HSPY = 'LF_cal_log_m_center_strict_peaks.hspy'
    ORG_HSPY = 'LeftFish_unmasked.hspy'
    FILE_KD = '220525_vector_match_sum_score_weighted_step05deg_distbound005.npy'
    FILE_ANG = '260525_vector_match_ang_score_step05deg_ang_thresh005.npy'
    FILE_SUM = '260525_vector_match_sum_score_step05deg.npy'
    FILE_INTENSITY = '270525_vector_match_score_intensity_step05deg_dist005.npy'

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

    
    frame = 56

    i, j = frame, frame+1


    ### Check overlay plot ###
    # experimental = np.load(DIR_NPY+'peaks_all_LoG.npy', allow_pickle=True)
    # method = "score_ang"
    # check_overlay_plot(experimental, frame, simulation, phase, orientation, method)
    
    ### EXPERIMENTAL ###
    exp_intensity = np.load(DIR_NPY+FILE_INTENSITY, allow_pickle=True)
    exp_intensity = to_orientation_map(exp_intensity,simulation)
    exp_weighted = np.load(DIR_NPY+FILE_KD, allow_pickle=True)
    exp_weighted = to_orientation_map(exp_weighted, simulation)
    exp_sum = np.load(DIR_NPY+FILE_SUM, allow_pickle=True)
    exp_sum = to_orientation_map(exp_sum, simulation)
    exp_ang = np.load(DIR_NPY+FILE_ANG, allow_pickle=True)
    exp_ang = to_orientation_map(exp_ang, simulation)

    # exp_results = exp_intensity


    ### Misorientation comparison ###
    # lbls = ('Score A', 'Score B', 'Score C', 'Score D')
    # clrs = ('Blue', 'Orange', 'Red', 'Green')
    # symbols = ('o', '^', 's', 'X')
    # datasets = [exp_sum, exp_weighted, exp_ang, exp_intensity]
    # plot.plot_compare_misorientation_scatter(datasets, lbls, clrs, symbols)
    # plot.plot_compare_misorientation_scatter(datasets, lbls, clrs, symbols,lim=True)
    # plot.plot_misorientation_violin(exp_weighted)
    

    ### Comparing score C to TM ###
    # first we get TM results
    frames = [10, 29, 56]
    for f in frames:
        plot.plot_with_markers(sim_results,DIR_HSPY+ORG_HSPY,f,f+1)
        plot.plot_ipf(sim_results, f, phase, orientation, cmap='viridis') # regular cmap not reversed cause score is opposite, get it?

    # misorientation comparison
    datasets = [exp_ang, sim_results]
    lbls = ('Score C', 'TM')
    clrs = ('Red', 'blue')
    symbols = ('s', 'o')
    plot.plot_compare_misorientation_scatter(datasets, lbls, clrs, symbols, lim=False, legend_loc='best' )
    plot.plot_compare_misorientation_scatter(datasets, lbls, clrs, symbols, lim=True, legend_loc='upper left')

    print("TM misorientation stats:")
    print(get_misorientation_statistics(sim_results))


    # print("Score A:")
    # print(get_misorientation_statistics(exp_sum))
    # print("Score B:")
    # print(get_misorientation_statistics(exp_weighted))
    # print("Score C:")
    # print(get_misorientation_statistics(exp_ang))
    # print("Score D:")
    # print(get_misorientation_statistics(exp_intensity))
   
    # print("exp:", exp_results.data[frame][0])
    # print("sim:", sim_results.data[frame][0])
    # plot.plot_misorientation_violin(exp_intensity)
    # plot.plot_ipf(exp_sum, frame, phase, orientation, cmap='viridis_r')
    # plot.plot_ipf(exp_weighted, frame, phase, orientation, cmap='viridis_r')
    # plot.plot_ipf(exp_ang, frame, phase, orientation, cmap='viridis_r')
    # plot.plot_ipf(exp_intensity, frame, phase, orientation, cmap='viridis_r')
    # plot.plot_with_markers(exp_sum, DIR_HSPY+ORG_HSPY, i, j)
    # plot.plot_with_markers(exp_weighted, DIR_HSPY+ORG_HSPY, i, j)
    # plot.plot_with_markers(exp_ang, DIR_HSPY+ORG_HSPY, i, j)
    # plot.plot_with_markers(exp_intensity, DIR_HSPY+ORG_HSPY, i, j)
    # plot.plot_ipf_all_best_orientations(exp_sum, phase, cmap='viridis_r')
    # plot.plot_ipf_all_best_orientations(exp_weighted, phase, cmap='viridis_r')
    # plot.plot_ipf_all_best_orientations(exp_ang, phase, cmap='viridis_r')
    # plot.plot_ipf_all_best_orientations(exp_intensity, phase, cmap='viridis_r')

    ### PLOTS ### 
    # plot.plot_ipf_all_best_orientations(exp_results, phase, cmap='viridis_r')
    # plot.plot_ipf_all_best_orientations(sim_results, phase, cmap='viridis_r')
    # plot.plot_misorientation_scatter(exp_results)
    # plot.plot_misorientation_scatter(sim_results)
    # plot.plot_ipf(sim_results,frame,phase,orientation, 'viridis')
    # plot.plot_ipf(exp_results,frame,phase,orientation, 'viridis_r')
    # plot.plot_with_markers(exp_results,DIR_HSPY+ORG_HSPY,i,j)
    # plot.plot_with_markers(sim_results,DIR_HSPY+ORG_HSPY,i,j)
    
