import hyperspy.api as hs
import pyxem as pxm
import numpy as np
import simulation as sim
import plotting as plot
import matplotlib.pyplot as plt
from sphere_matching import vector_match, fast_polar
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
    n_array = vector_match([exp[frame]], sim,step_size=0.5, reciprocal_radius=1.35, n_best=1)
    t2 = time()
    print(n_array.shape)
    n_best = n_array[0][0]
    frame, score, rotation, mirror = n_best[0],n_best[1],n_best[2], n_best[3]
    print(f"Computation time: {(t2-t1)/60} min")
    print('Best frame:', frame)
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
    FILE = '060525ormap_step05deg_vector_match_ang_score_NO_MIRROR_wrap_degrees_v060525.npy'

    s = hs.load(DIR_HSPY+HSPY)
    ### UNCOMMENT FOR crystal_map ### 
    # ------------------------------------------ #
    # s = np.reshape(s.data,(6,10,256,256))
    # s = pxm.signals.ElectronDiffraction2D(s)
    # s.set_diffraction_calibration(0.0107)
    # print(s.axes_manager.navigation_axes[0].units)
    # print(s.axes_manager)
    # s.axes_manager.navigation_axes[0].units = r"$Å^{-1}$"
    # print(s.axes_manager.navigation_axes[0].units)
    # ------------------------------------------ #


    ### SIMULATED ###
    s_pol = s.get_azimuthal_integral2d(npt=112, radial_range=(0.,1.35))
    phase = sim.unit_cell()
    grid, orientation = sim.gen_orientation_grid(phase)
    simgen = sim.get_simulation_generator(precession_angle=1., minimum_intensity=1e-4, approximate_precession=True)
    simulation = sim.compute_simulations(simgen, phase, grid, reciprocal_radius=1.35,
                                      max_excitation_error=0.05)

    # sim_results = s_pol.get_orientation(simulation,n_best=grid.size,frac_keep=1.)  # Creates an OrientationMap

    ### Test check_overlay_plot ### 
    FILE_STRICT = 'peaks_all_LoG.npy'
    FILE_SIM = 'LF_r_theta_sim.npy'
    experimental = np.load(DIR_NPY+FILE_STRICT,allow_pickle=True)
    # Quick polar transform
    experimental = fast_polar(experimental)
    simulated = np.load(DIR_NPY+FILE_SIM,allow_pickle=True)

    frame = 56
    check_overlay_plot(experimental,frame, simulated, simulation)


    i, j = frame, frame+1
    
    ### EXPERIMENTAL ###
    # exp_results = np.load(DIR_NPY+FILE, allow_pickle=True)
    # exp_results = to_orientation_map(exp_results,simulation)

    

    # stats = compare_orientations(sim_results.data, exp_results.data)
    # print(stats)

    # exp_results.data[frame][0][2] += 107.64028464817187 + 90
    # exp_results.data[frame][0][3] = 0
    # exp_results.data[frame][0][0] = 439
    # sim_results.data[frame][0][0] = exp_results.data[frame][0][0]
    ### PLOTS ### 
    # plot.plot_misorientation_scatter(exp_results)
    # plot.plot_misorientation_scatter(sim_results)
    # plot.plot_ipf(sim_results,frame,phase,orientation, 'viridis')
    # plot.plot_ipf(exp_results,frame,phase,orientation, 'viridis_r')
    # plot.plot_with_markers(sim_results,DIR_HSPY+ORG_HSPY,i,j)
    # plot.plot_with_markers(exp_results,DIR_HSPY+ORG_HSPY,i,j)
    #
    ### Crystal map (reshaped dataset) ### 
    # plot.plot_crystal_map(sim_results,phase)

    
