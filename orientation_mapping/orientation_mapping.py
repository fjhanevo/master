import hyperspy.api as hs
import pyxem as pxm
import orix 
from diffpy.structure import Atom, Lattice, Structure
from diffsims.generators.simulation_generator import SimulationGenerator
import matplotlib.pyplot as plt
import numpy as np

def unit_cell(a=4.0495):
    atoms = [Atom('Al', [0,0,0]), Atom('Al', [0.5,0.5,0]),
             Atom('Al', [0.5, 0, 0.5]), Atom('Al', [0,0.5,0.5])]

    lattice = Lattice(a,a,a,90,90,90)
    phase = orix.crystal_map.Phase(name='Al', space_group=225, structure=Structure(atoms, lattice))
    return phase

def gen_orientation_grid(phase, angular_resolution=0.5):
    grid = orix.sampling.get_sample_reduced_fundamental(
        angular_resolution,
        point_group=phase.point_group,
    )

    orientations = orix.quaternion.Orientation(grid, symmetry=phase.point_group)
    orientations.scatter('ipf')
    
    return grid, orientations

def get_simulation_generator(precession_angle=1.0, 
                             minimum_intensity=1e-4,
                             approximate_precession=True):

    return SimulationGenerator(precession_angle=precession_angle,
                               minimum_intensity=minimum_intensity,
                               approximate_precession=approximate_precession)

def compute_simulations(simgen,phase,grid,reciprocal_radius=1.35, 
                        max_excitation_error=0.01,with_direct_bream=False):
    return simgen.calculate_diffraction2d(
        phase=phase,                                 # phase to simulate for
        rotation=grid,                               # orientations to simulate for
        reciprocal_radius=reciprocal_radius,         # max radius to consider in [Å^-1]
        with_direct_beam=with_direct_bream,          # option to include direct beam
        max_excitation_error=max_excitation_error,   # max excitation error, s
    )

def make_ipf(sim, idx, phase, orientation):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='ipf', symmetry=phase.point_group)

    correlations = sim.inav[idx].data[:,1]
    tm_indices = (sim.inav[idx].data[:,0]).astype('int16')
    orientations = orientation[tm_indices]
    loris= sim.to_single_phase_orientations()
    loris_best = loris[idx,0]
    ax.scatter(orientations, c=correlations, cmap='viridis')
    ax.scatter(loris_best,c='red',marker='o',s=100)
    plt.show()


if __name__ == '__main__':
    DIR_NPY = 'npy_files/'
    DIR_HSPY = 'processed_hspy_files/'
    FILE = 'test_vector_match_ang0005.npy'
    HSPY = 'LF_cal_log_m_center_m_peaks.hspy'
    ORG_HSPY = 'LeftFish_unmasked.hspy'

    ### SIMULATED ###
    s = hs.load(DIR_HSPY+HSPY)
    s_pol = s.get_azimuthal_integral2d(npt=112, radial_range=(0.,1.35))
    phase = unit_cell()
    grid, orientation = gen_orientation_grid(phase)
    simgen = get_simulation_generator(precession_angle=1., minimum_intensity=1e-4, approximate_precession=True)
    simulations = compute_simulations(simgen, phase, grid, reciprocal_radius=1.35,
                                      max_excitation_error=0.05)

    results = s_pol.get_orientation(simulations,n_best=1,frac_keep=1.)  # Creates an OrientationMap

    ### EXPERIMENTAL ###
    exp_results = np.load(DIR_NPY+FILE, allow_pickle=True)
    # print(type(exp_results))
    exp_results = pxm.signals.indexation_results.OrientationMap(exp_results)
    # print(type(exp_results))
    # print(exp_results.metadata)
    exp_results.metadata.VectorMetadata.column_names = ['index', 'correlation', 'rotation', 'factor']
    exp_results.metadata.VectorMetadata.units= ['a.u', 'a.u', 'deg', 'a.u']
    # Manually assign new simulation VERY IMPORTANT OR IT WILL BE ANGRY
    exp_results.simulation = simulations 
    make_ipf(exp_results, 56, phase, orientation)
    # make_ipf(results, 56, phase, orientation)

    #
    i,j = 54, 58
    s_org = hs.load(DIR_HSPY+ORG_HSPY)
    s_org = s_org.inav[i:j]
    exp_results = exp_results.inav[i:j]
    s_org.plot(cmap='viridis_r', norm='log',title='', colorbar=False, scalebar_color='black',axes_ticks='off')
    s_org.add_marker(exp_results.to_markers(annotate=True))
    plt.show()
