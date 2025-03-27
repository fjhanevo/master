import hyperspy.api as hs
import pyxem as pxm
import orix 
from diffpy.structure import Atom, Lattice, Structure
from diffsims.generators.simulation_generator import SimulationGenerator
import diffsims
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


def get_polar_homies(sim,filename:str) -> None:
    # Get simulations and skip intensities
    r,theta,_ = sim.polar_flatten_simulations()
    # Combine to a (M,Q,2) ndarray
    rt = np.stack([r,theta], axis=-1)
    # save that B
    np.save(file=filename, arr=rt, allow_pickle=True)
        
def save_simulation(sim, filename:str) -> None:
    np.save(file=filename, arr=sim.data, allow_pickle=True)
        
if __name__ == '__main__':
    DIR_HSPY = 'processed_hspy_files/' 
    DIR_NPY = 'npy_files/'
    FILE = 'LeftFish_unmasked.hspy'
    # FILE = 'LeftFish_masked_log_calibrated.hspy'
    SIM_FILE = 'test_orientation_map.npy'

    # Load the file
    s = hs.load(DIR_HSPY+FILE)

    # print(s.axes_manager)
    # print(s.data.shape)
    # s = np.reshape(s.data, (6,10,256,256))
    # print(s.data.shape)
    # s = pxm.signals.ElectronDiffraction2D(s)
    # s.set_diffraction_calibration(0.0107)
    # print(s.axes_manager)
    # s.plot(cmap='viridis_r')
    # plt.show()
    s_pol = s.get_azimuthal_integral2d(npt=112, radial_range=(0.,1.35))
    phase = unit_cell()
    grid, orientation = gen_orientation_grid(phase)
    simgen = get_simulation_generator(precession_angle=1., minimum_intensity=1e-4, approximate_precession=True)
    simulations = compute_simulations(simgen, phase, grid, reciprocal_radius=1.35,
                                      max_excitation_error=0.05)

    results = s_pol.get_orientation(simulations,n_best=1,frac_keep=1.)  # Creates an OrientationMap
    # print(results.metadata)
    print(results.simulation._phase_slider)
    # results = results.data.copy()      # Make it to a ndarray
    # results = pxm.signals.indexation_results.OrientationMap(results)    # Back to OrientationMap
    # print(type(results))
    # print(results.simulation)   # returns None
    # make_ipf(results,56,phase, orientation)

    # results = np.load(DIR_NPY+SIM_FILE, allow_pickle=True)
    # results = pxm.signals.indexation_results.OrientationMap(results)

    #### DETTE ER NYTTIG!!
    # xmap = results.to_crystal_map()
    # print(type(xmap))
    # oris = xmap.orientations[56,:]
    #
    # oris.to_euler()
    # print(oris.data.shape)
    # print(oris.to_euler())
