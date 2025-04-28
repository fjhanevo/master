import hyperspy.api as hs
import pyxem as pxm
import numpy as np
import simulation as sim
import plotting as plot


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

if __name__ == '__main__':
    DIR_NPY = 'npy_files/'
    DIR_HSPY = 'processed_hspy_files/'
    HSPY = 'LF_cal_log_m_center_strict_peaks.hspy'
    ORG_HSPY = 'LeftFish_unmasked.hspy'
    FILE = 'ormap_step05deg_dist005_penalty075.npy'

    s = hs.load(DIR_HSPY+HSPY)
    ### UNCOMMENTED FOR crystal_map ### 
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

    sim_results = s_pol.get_orientation(simulation,n_best=grid.size,frac_keep=1.)  # Creates an OrientationMap
    frame = 29
    i, j = frame, frame+1
    
    ### EXPERIMENTAL ###
    exp_results = np.load(DIR_NPY+FILE, allow_pickle=True)
    print(exp_results.shape)
    exp_results = to_orientation_map(exp_results,simulation)

    ### PLOTS ### 
    plot.plot_misorientation_scatter(exp_results)
    plot.plot_ipf(sim_results,frame,phase,orientation, 'viridis')
    plot.plot_ipf(exp_results,frame,phase,orientation, 'viridis_r')
    plot.plot_with_markers(sim_results,DIR_HSPY+ORG_HSPY,i,j)
    plot.plot_with_markers(exp_results,DIR_HSPY+ORG_HSPY,i,j)

    ### Crystal map (reshaped dataset) ### 
    # plot.plot_crystal_map(sim_results,phase)

    
