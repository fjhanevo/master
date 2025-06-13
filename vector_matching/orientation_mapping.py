from sys import maxsize
import hyperspy.api as hs
import pyxem as pxm
import numpy as np
import simulation as sim
import plotting as plot
import matplotlib.pyplot as plt
import vector_match as vm
from vm_utils import fast_polar
from time import time

def to_orientation_map(data, simulation):
    """
    Converts numpy array into OrientationMap and adds metdata.
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
    exp = fast_polar(exp)
    if method=="score_intensity":
        sim_data = np.load(DIR_NPY+'sim_r_theta_intensity.npy', allow_pickle=True)
        # exp = fast_polar(exp[:,:2])
    else:
        sim_data = np.load(DIR_NPY+'LF_r_theta_sim.npy', allow_pickle=True)
        # exp = fast_polar(exp)
    t1 = time()
    sim_trimmed = sim_data[:1200]   # did this to decrease comp time, as I know the answer
    n_array = vm.vector_match(
        [exp[frame]], 
        sim_data,
        step_size=0.5, 
        reciprocal_radius=1.35, 
        n_best=len(sim_data), 
        method=method,
    )

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

def get_normalised_misorientation_statistics(data):
    loris = data.to_single_phase_orientations()
    loris_best = loris[:, 0]
    loris_ang = loris_best.angle_with_outer(loris_best, degrees=True)
    num_frames = loris_ang.shape[0]

    all_normalised = []
    stats = []

    for i in range(num_frames):
        norm_mis = []
        for j in range(num_frames):
            if i != j:
                mis = loris_ang[i, j]
                delta = np.abs(i-j)
                norm_mis.append(mis / delta)
        norm_mis = np.array(norm_mis)

        stats.append({
            "mean": np.mean(norm_mis),
            "std": np.std(norm_mis),
            "min": np.min(norm_mis),
            "max": np.max(norm_mis),
            "median": np.median(norm_mis),
        })

        all_normalised.extend(norm_mis)

    all_normalised = np.array(all_normalised)
    normalised_stats = {
        "norm_mean" : np.mean(all_normalised),
        "norm_std" : np.std(all_normalised),
        "norm_min" : np.min(all_normalised),
        "norm_max" : np.max(all_normalised),
        "norm_median" : np.median(all_normalised),
    }

    return normalised_stats


def get_algorithm_accuracy_statistics(data):
    data = data.data if hasattr(data, "data") else data
    # sim_data = np.load(DIR_NPY+sim_file, allow_pickle=True) 
    best_scores = data[:, 0, 1]

    stats = {
        "mean_score" : np.mean(best_scores),
        "min_score" : np.min(best_scores),
        "max_score" : np.max(best_scores),
        "std_score" : np.std(best_scores),
        "num_matched" : np.sum(best_scores==0.0),
        "percent_matched" : 100.0 * np.sum(best_scores==0.0) / best_scores.size,
    }

    return stats

def print_n_scores(data, frame, n):
    for i in range(n):
        print(data.data[frame][i][1])
     
def get_score_uniqueness(data, frame, n, threshold: float = 1e-4):
    scores = data.data if hasattr(data, "data") else data
    top_scores = [scores[frame][i][1] for i in range(n)]
    top_scores = np.array(top_scores)

    best_score = top_scores[0]
    max_score =top_scores[-1]

    close_count = np.sum(np.abs(top_scores - best_score) < threshold)

    stats = {
        'std': np.std(top_scores),
        'range_ratio': np.abs(max_score - best_score) / best_score if best_score > 0 else 0.0,
        'percent_close_to_best': 100 * close_count / n
    }

    return stats


if __name__ == '__main__':
    DIR_NPY = 'npy_files/'
    DIR_HSPY = 'processed_hspy_files/'
    HSPY = 'LF_cal_log_m_center_strict_peaks.hspy'
    ORG_HSPY = 'LeftFish_unmasked.hspy'
    FILE_KD = '220525_vector_match_sum_score_weighted_step05deg_distbound005.npy'
    FILE_ANG = '260525_vector_match_ang_score_step05deg_ang_thresh005.npy'
    FILE_SUM = '260525_vector_match_sum_score_step05deg.npy'
    FILE_INTENSITY = '030625_vector_match_score_intensity_step05deg_dist005.npy'
    FILE_ANG_ACCURACY = '280525_VM_ang_score_algorithm_accuracy.npy'

    hs.set_log_level('WARNING')
    s = hs.load(DIR_HSPY+HSPY)
    
    ### SIMULATED ###
    s_pol = s.get_azimuthal_integral2d(npt=112, radial_range=(0.,1.35))
    phase = sim.unit_cell()
    grid, orientation = sim.gen_orientation_grid(phase)
    simgen = sim.get_simulation_generator(precession_angle=1., minimum_intensity=1e-4, approximate_precession=True)
    simulation = sim.compute_simulations(simgen, phase, grid, reciprocal_radius=1.35,
                                      max_excitation_error=0.05)
    #
    sim_results = s_pol.get_orientation(simulation,n_best=grid.size,frac_keep=1.)  # Creates an OrientationMap

    
    frame =56 

    i, j = frame, frame+1


    ### Check overlay plot ###
    """POLAR TRANSFORMERING GJØRES I check_overlay_plot!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111"""
    # experimental = np.load(DIR_NPY+'peaks_all_LoG.npy', allow_pickle=True)
    # exp_intensity = np.load(DIR_NPY+'peaks_intensity_all_LoG.npy',allow_pickle=True)
    # FYYYYYYYYY FAAAAAAAAAAAAEEEEEEEEEEEEENNNNNNNNNNNNNNN 
    # experimental_peaks_bad = np.load(DIR_NPY+'310525_liberal_peaks_for_discussion_LoG.npy', allow_pickle=True)
    # experimental_peaks_worse = np.load(DIR_NPY+'030625_more_liberal_peaks_for_discussion_LoG.npy', allow_pickle=True)
    # method = "score_ang"
    # check_overlay_plot(experimental, frame, simulation, phase, orientation, method)
    # method = "score_intensity"
    # check_overlay_plot(exp_intensity, frame, simulation, phase, orientation, method)
    # check_overlay_plot(experimental_peaks_bad, frame, simulation, phase, orientation, method)
    # check_overlay_plot(experimental_peaks_bad, 29, simulation, phase, orientation, method)
    # check_overlay_plot(experimental_peaks_worse, 29, simulation, phase, orientation, method)
    
    ### EXPERIMENTAL ###
    exp_intensity = np.load(DIR_NPY+FILE_INTENSITY, allow_pickle=True)
    exp_intensity = to_orientation_map(exp_intensity,simulation)
    exp_weighted = np.load(DIR_NPY+FILE_KD, allow_pickle=True)
    exp_weighted = to_orientation_map(exp_weighted, simulation)
    exp_sum = np.load(DIR_NPY+FILE_SUM, allow_pickle=True)
    exp_sum = to_orientation_map(exp_sum, simulation)
    exp_ang = np.load(DIR_NPY+FILE_ANG, allow_pickle=True)
    exp_ang = to_orientation_map(exp_ang, simulation)

    n = 20
    # print_n_scores(exp_weighted, 56, n)
    print(get_score_uniqueness(sim_results, 58, n))
    print(get_score_uniqueness(exp_weighted, 58, n))
    # print(get_score_uniqueness(sim_results, 56, n))
    # print("Score A:")
    # print(get_normalised_misorientation_statistics(exp_sum))
    # print("Score B:")
    # print(get_normalised_misorientation_statistics(exp_weighted))
    # print("Score C:")
    # print(get_normalised_misorientation_statistics(exp_ang))
    # print("Score D:")
    # print(get_normalised_misorientation_statistics(exp_intensity))


    # frames = [0,1,2,3,4,5]
    # colors=['black', 'blue', 'red', 'green', 'yellow', 'brown']
    # plot.plot_ipf_all_best_orientations_subset(exp_ang, phase, frames, colors)


    # exp_ang_accuracy = np.load(DIR_NPY+FILE_ANG_ACCURACY, allow_pickle=True)
    # exp_ang_accuracy_reshaped = np.reshape(exp_ang_accuracy, (14,299))
    # exp_ang_accuracy = to_orientation_map(exp_ang_accuracy, simulation)
    # plot.test_plot_ipf(exp_ang_accuracy, phase, orientation,'magma')
    # plot.plot_ipf_all_best_orientations(exp_ang_accuracy, phase, cmap='viridis_r')
    # print(get_algorithm_accuracy_statistics(exp_ang_accuracy))

    ### Misorientation comparison ###
    # lbls = ('Score A', 'Score B', 'Score C', 'Score D')
    # clrs = ('Blue', 'Orange', 'Red', 'Green')
    # symbols = ('o', 's', 'X', '^')
    # datasets = [exp_sum, exp_weighted, exp_ang, exp_intensity]
    # plot.plot_compare_misorientation_scatter(datasets, lbls, clrs, symbols,legend_loc='upper left')
    # plot.plot_compare_misorientation_scatter(datasets, lbls, clrs, symbols,lim=True,legend_loc='upper left')
    # plot.plot_misorientation_violin(exp_sum)
    # plot.plot_misorientation_violin(exp_weighted)
    # plot.plot_misorientation_violin(exp_ang)
    # plot.plot_misorientation_violin(exp_intensity)
    # plot.plot_misorientation_violin(sim_results)

    ### Comparing score C to TM ###
    # first we get TM results
    # frames = [10, 29, 56]
    # for f in frames:
    #     plot.plot_with_markers(sim_results,DIR_HSPY+ORG_HSPY,f,f+1)
    #     plot.plot_ipf(sim_results, f, phase, orientation, cmap='viridis') # regular cmap not reversed cause score is opposite, get it?

    # misorientation comparison
    # datasets = [exp_ang, sim_results]
    # lbls = ('Score C', 'TM')
    # clrs = ('Red', 'blue')
    # symbols = ('X', 'o')
    # plot.plot_compare_misorientation_scatter(datasets, lbls, clrs, symbols, lim=False, legend_loc='best' )
    # plot.plot_compare_misorientation_scatter(datasets, lbls, clrs, symbols, lim=True, legend_loc='upper left')
    # print("TM misorientation stats:")
    # print(get_misorientation_statistics(sim_results))

    # plot.plot_ipf_all_best_orientations(sim_results, phase, cmap='viridis_r')


    # print("Score A:")
    # print(get_misorientation_statistics(exp_sum))
    # print("Score B:")
    # print(get_misorientation_statistics(exp_weighted))
    # print("Score C:")
    # print(get_misorientation_statistics(exp_ang))
    # print("Score D:")
    # print(get_misorientation_statistics(exp_intensity))
   

    # print("sim:", sim_results.data[frame][0])
    # plot.plot_misorientation_violin(exp_intensity)
    # plot.plot_ipf(exp_sum, frame, phase, orientation, cmap='viridis_r')
    # plot.plot_ipf(exp_weighted, frame, phase, orientation, cmap='viridis_r')
    # plot.plot_ipf(exp_ang, frame, phase, orientation, cmap='viridis_r')
    # plot.plot_ipf(exp_intensity, frame, phase, orientation, cmap='viridis_r')
    # plot.plot_with_markers(exp_sum, DIR_HSPY+ORG_HSPY, i, j)
    # plot.plot_with_markers(exp_weighted, DIR_HSPY+ORG_HSPY, i, j)
    # i, j = 40, 50
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
    
