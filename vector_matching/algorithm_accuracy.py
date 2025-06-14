import pyxem as pxm
import orix
from orix.vector import Vector3d
import numpy as np
import matplotlib.pyplot as plt
import simulation as sim
from orientation_mapping import to_orientation_map

def get_tm_accuracy_simulation(simulation, calibration, shape):
    sim_dps= []

    for i in range(0, simulation.current_size):
        simulation.rotation_index = i
        sim_dps += [(simulation.get_diffraction_pattern(shape=shape, sigma=1, calibration=calibration))]
    # reshape
    sim_dps = np.reshape(sim_dps, shape)
    sim_dps = pxm.signals.ElectronDiffraction2D(sim_dps)
    sim_dps.set_diffraction_calibration(calibration)

    return sim_dps

def plot_ipf_w_crystal_map(data, phase):

    xmap = data.to_crystal_map()
    oris = xmap.orientations
    print(oris)
    fig = plt.figure()
    v = Vector3d.zvector()
    key_z = orix.plot.IPFColorKeyTSL(phase.point_group, v)
    ax = fig.add_subplot(111, projection='ipf', symmetry=phase.point_group, direction=v)
    ax.scatter(oris, c=key_z.orientation2color(oris), cmap='viridis_r', s=10)
    plt.show()

    ### Correlation score plot ###
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='ipf', symmetry=phase.point_group, direction=v)
    corrs = data.data[:,:,0,1].flatten()
    ax.scatter(oris, c=corrs, cmap='viridis_r', s=10)
    plt.show()
if __name__ == '__main__':
    DIR_NPY = 'npy_files/'
    DIR_HSPY = 'processed_hspy_files/'
    HSPY = 'LF_cal_log_m_center_strict_peaks.hspy'
    ORG_HSPY = 'LeftFish_unmasked.hspy'
    FILE_ANG_ACCURACY = '280525_VM_ang_score_algorithm_accuracy.npy'

    ### Parameters ###
    calibration = 0.0107
    reciprocal_radius = 1.35
    new_shape = (14, 299, 4186, 4)  # reshape for cool crystal map plot


    ### SIMULATED ###
    phase = sim.unit_cell()
    grid, orientation = sim.gen_orientation_grid(phase)
    simgen = sim.get_simulation_generator(precession_angle=1., minimum_intensity=1e-4, approximate_precession=True)
    simulation = sim.compute_simulations(simgen, phase, grid, reciprocal_radius=reciprocal_radius,
                                      max_excitation_error=0.05)

    sim_dps= get_tm_accuracy_simulation(simulation, new_shape[2:],calibration)
    s_pol = sim_dps.get_azimuthal_integral2d(npt=112, radial_range=(0., reciprocal_radius))
    sim_results = s_pol.get_orientation(simulation, n_best=grid.size, frac_keep=1.0)

    ### Experimental ###
    exp_ang = np.load(DIR_NPY+FILE_ANG_ACCURACY, allow_pickle=True)
    exp_ang = np.reshape(exp_ang,new_shape)
    exp_ang = to_orientation_map(exp_ang, simulation)

    ### Plotting ###

    plot_ipf_w_crystal_map(exp_ang, phase)
