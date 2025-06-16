import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np

import simulation as sim
from vector_match import vector_match
from orientation_mapping import to_orientation_map
import plotting as plot


def create_file() -> None:
    # ag_peaks = np.load(DIR_NPY+'160625_Ag_peaks.npy', allow_pickle=True)
    ag_peaks = np.load(DIR_NPY+'160625_Ag_peaks_intensity.npy', allow_pickle=True)
    ag_sim_peaks = np.load(DIR_NPY+'160625_sim_ag_intensity.npy', allow_pickle=True)
    # print(f'Peaks shape: {ag_peaks.shape}')
    # print(f'Sim shape: {ag_sim_peaks.shape}')
    #
    ag_score = vector_match(
        experimental=ag_peaks,
        simulated=ag_sim_peaks,
        step_size=step_size,
        reciprocal_radius=reciprocal_radius,
        n_best=len(ag_sim_peaks),
        method='score_intensity'
    )
    #
    np.save(file=DIR_NPY+'160625_AG_scoreD.npy', arr=ag_score, allow_pickle=True)


if __name__ == '__main__':
    DIR_HSPY = 'processed_hspy_files/'
    DIR_NPY = 'npy_files/'
    HSPY_ORG = 'SPED_Ag_labels_sum_signal.hspy'
    FILE_HSPY = '160625_SPED_ag_no_background.hspy'

    dp = hs.load(DIR_HSPY+FILE_HSPY)
    # dp.plot(cmap='magma_r',norm='log',title='',colorbar=False,
    #          scalebar=True,scalebar_color='black', axes_ticks='off')
    # plt.show()

    ### params ###
    reciprocal_radius = 1.5
    step_size = 0.5 # degs

    ### Simulation ####
    phase = sim.unit_cell(a=4.08, name='Ag')
    grid, orientation = sim.gen_orientation_grid(phase)
    simgen = sim.get_simulation_generator(
        precession_angle=1.,
        minimum_intensity=1e-5,
        approximate_precession=True
    )

    simulation = sim.compute_simulations(
        simgen, phase, grid, reciprocal_radius=reciprocal_radius, max_excitation_error=0.01
    )

    s_pol = dp.get_azimuthal_integral2d(npt=112, radial_range=(0.,reciprocal_radius))
    tm = s_pol.get_orientation(simulation,n_best=grid.size,frac_keep=1.)  # Creates an OrientationMap

    # filename = '160625_sim_ag_intensity.npy'
    # sim.get_polar_coords(simulation, DIR_NPY+filename)



    
    frame = 2
    
    score = np.load(DIR_NPY+'160625_AG_scoreD.npy', allow_pickle=True)
    score = to_orientation_map(score, simulation)
    plot.plot_misorientation_scatter(score)
    plot.plot_misorientation_scatter(tm)
    plot.plot_ipf(score, frame, phase, orientation,cmap='viridis_r')
    plot.plot_ipf(tm, frame, phase, orientation,cmap='viridis')
    plot.plot_with_markers(score, DIR_HSPY+HSPY_ORG, frame,frame+1)
    plot.plot_with_markers(tm, DIR_HSPY+HSPY_ORG, frame,frame+1)

    

