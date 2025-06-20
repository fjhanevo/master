import orix 
from diffpy.structure import Atom, Lattice, Structure
from diffsims.generators.simulation_generator import SimulationGenerator
import numpy as np

"""
File to create a simulation of FCC Al templates.
"""

def unit_cell(a=4.0495,name='Al'):
    atoms = [Atom(name, [0,0,0]), Atom(name, [0.5,0.5,0]),
             Atom(name, [0.5, 0, 0.5]), Atom(name, [0,0.5,0.5])]

    lattice = Lattice(a,a,a,90,90,90)
    phase = orix.crystal_map.Phase(name=name, space_group=225, structure=Structure(atoms, lattice))
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

def get_polar_coords(sim,filename:str) -> None:
    # Get simulations and skip intensities
    r,theta,_ = sim.polar_flatten_simulations()
    # Combine to a (M,Q,2) ndarray
    rt = np.stack([r,theta], axis=-1)
    # save that B
    np.save(file=filename, arr=rt, allow_pickle=True)
