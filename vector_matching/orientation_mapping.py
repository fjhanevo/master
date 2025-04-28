import hyperspy.api as hs
import pyxem as pxm
from orix.plot import IPFColorKeyTSL
from orix.vector import Vector3d
import matplotlib.pyplot as plt
import simulation as sim

def plot_ipf(data, idx, phase, orientation ,cmap:str):
    """
    Plots an IPF with a red marker indicating the best found orientation.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='ipf', symmetry=phase.point_group)

    correlations = data.inav[idx].data[:,1]
    tm_indices = (data.inav[idx].data[:,0]).astype('int16')
    orientations = orientation[tm_indices]
    loris= data.to_single_phase_orientations()
    loris_best = loris[idx,0]
    ax.scatter(orientations, c=correlations, cmap=cmap)
    ax.scatter(loris_best,c='red',marker='o',s=100) # best found orientation
    plt.show()

def plot_with_markers(results, file,i,j):
    """
    Plot markers on dataset
    """
    data = hs.load(file)
    data = data.inav[i:j]
    results = results.inav[i:j]
    data.plot(cmap='viridis_r', norm='log', title='', colorbar=False, scalebar_color='black', axes_ticks='off')
    data.add_marker(results.to_markers(annotate=True))
    plt.show()

def plot_misorientation_hist(data):
    loris = data.to_single_phase_orientations()
    loris_best = loris[:,0]
    loris_ang  = loris_best.angle_with_outer(loris_best, degrees=True)
    plt.figure()
    plt.hist(loris_ang.flatten(), bins=60)
    plt.xlabel('Degrees')
    plt.ylabel('Count')
    plt.show()

def plot_misorientation_scatter(data):
    loris = data.to_single_phase_orientations()
    loris_best = loris[:,0]
    loris_ang = loris_best.angle_with_outer(loris_best,degrees=True)

    plt.figure(figsize=(8,6))
    for i in range(len(loris_ang)-1):
        plt.scatter(i, loris_ang[i,i+1], s=34,c='black')

    plt.axhline(y = 1, color='red', label=r'1$\degree$', linestyle='dashed')
    plt.grid(True)
    plt.ylabel(r'Misorientation$\degree$',fontsize='26')
    plt.xlabel('Tilt Step',fontsize='26')
    plt.xticks(fontsize='18')
    plt.yticks(fontsize='18')
    plt.legend(fontsize='18', loc='center left')
    plt.tight_layout()
    plt.show()

def plot_crystal_map(results,phase):
    """
    Assumes reshaped dataset!
    """
    xmap = results.to_crystal_map()
    oris = xmap.orientations
    corrs = results.data[:,:,0,1].flatten()
    print(corrs.shape)

    key_x = IPFColorKeyTSL(phase.point_group, Vector3d.xvector())
    key_y = IPFColorKeyTSL(phase.point_group, Vector3d.yvector())
    key_z = IPFColorKeyTSL(phase.point_group, Vector3d.zvector())

    oris_z = key_z.orientation2color(oris)[:,:]
    xmap.plot(oris_z, overlay=corrs, remove_padding=True)
    plt.show()
    oris_x = key_x.orientation2color(oris)[:,:]
    xmap.plot(oris_x, overlay=corrs, remove_padding=True)
    plt.show()
    oris_y = key_y.orientation2color(oris)[:,:]
    xmap.plot(oris_y, overlay=corrs, remove_padding=True)
    plt.show()


def to_orientation_map(data, simulation):
    """
    Converts numpy array into OrientaionMap and adds metdata
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
    
    print(sim_results.data.shape)
    ### EXPERIMENTAL ###
    # exp_results = np.load(DIR_NPY+FILE, allow_pickle=True)
    # exp2 = np.load(DIR_NPY+'ormap_step1deg_dist005_penalty075.npy', allow_pickle=True)
    # exp2 = to_orientation_map(exp2,simulation)
    # print(exp_results.shape)
    # exp_results = to_orientation_map(exp_results,simulation)
    # exp_reshaped = np.reshape(exp_results, (6,10))

    ### PLOTS ### 
    # plot_misorientation_scatter(exp_results)
    # plot_misorientation_scatter(exp2)
    # plot_ipf(sim_results,frame,phase,orientation, 'viridis')
    # plot_ipf(exp_results,frame,phase,orientation, 'viridis_r')
    # plot_with_markers(sim_results,DIR_HSPY+ORG_HSPY,i,j)
    # plot_with_markers(exp_results,DIR_HSPY+ORG_HSPY,i,j)
    # plot_misorientation(exp_results)
    # plot_misorientation(sim_results)

    # plot_crystal_map(sim_results,phase)

    
