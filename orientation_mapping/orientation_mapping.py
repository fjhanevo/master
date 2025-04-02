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

def plot_ipf(data, idx, phase, orientation):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='ipf', symmetry=phase.point_group)

    correlations = data.inav[idx].data[:,1]
    tm_indices = (data.inav[idx].data[:,0]).astype('int16')
    orientations = orientation[tm_indices]
    loris= data.to_single_phase_orientations()
    loris_best = loris[idx,0]
    ax.scatter(orientations, c=correlations, cmap='viridis_r')
    ax.scatter(loris_best,c='red',marker='o',s=100)
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
    HSPY = 'LF_cal_log_m_center_m_peaks.hspy'
    ORG_HSPY = 'LeftFish_unmasked.hspy'
    FILE = 'ormap_strict_rad2deg_n_best_all_ang_step0005.npy'

    ### SIMULATED ###
    s = hs.load(DIR_HSPY+HSPY)
    s_pol = s.get_azimuthal_integral2d(npt=112, radial_range=(0.,1.35))
    phase = unit_cell()
    grid, orientation = gen_orientation_grid(phase)
    simgen = get_simulation_generator(precession_angle=1., minimum_intensity=1e-4, approximate_precession=True)
    simulation = compute_simulations(simgen, phase, grid, reciprocal_radius=1.35,
                                      max_excitation_error=0.05)

    sim_results = s_pol.get_orientation(simulation,n_best=grid.size,frac_keep=1.)  # Creates an OrientationMap
    frame = 56 

    ### EXPERIMENTAL ###
    exp_results = np.load(DIR_NPY+FILE, allow_pickle=True)
    exp_results = to_orientation_map(exp_results,simulation)

    plot_ipf(exp_results,frame,phase,orientation)
