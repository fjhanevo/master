import numpy as np
import pyxem as pxm
from scipy.spatial import cKDTree
from sphere_matching import vector_to_3D, filter_sim, apply_z_rotation, wrap_degrees, vector_match_ang_score
from time import time
import plotting
import simulation as sim

def algorithm_accuracy(data, stepsize, reciprocal_radius, n_best):
    """
    Use VM and TM to match with itself.
    """
    tm_data_to_obj = pxm.signals.Diffraction2D(data)
    s_pol = tm_data_to_obj.get_azimuthal_integral2d(npt=112, radial_range=(0.,1.35))
    phase = sim.unit_cell()
    grid, orientation = sim.gen_orientation_grid(phase)

    simgen = sim.get_simulation_generator(precession_angle=1., minimum_intensity=1e-4, approximate_precession=True)
    simulation = sim.compute_simulations(simgen, phase, grid, reciprocal_radius=1.35,
                                      max_excitation_error=0.05)

    ### This hopefully template matches itself
    template_matching = s_pol.get_orientation(simulation,n_best=n_best,frac_keep=1.) 

    vector_matching = vector_match_ang_score(data, data, stepsize,reciprocal_radius,n_best) 
    


#NOTE: Slett denne
def vector_match_score_test(
    experimental:np.ndarray, 
    simulated:np.ndarray, 
    step_size:float, 
    reciprocal_radius:float,
    distance_bound=0.05,
    unmatched_penalty=1.0
) -> np.ndarray:
    """
    Match only one frame
    """

    step_size_rad = np.deg2rad(step_size)
    result_lst = []
    t1 = time()
    precomputed_trees = [
        [cKDTree(rot_frame) for rot_frame in filter_sim(sim_frame, step_size_rad, reciprocal_radius)]
        for sim_frame in simulated
    ]
    t2 = time()
    print(f"Pre-compute time: {(t2-t1)} sec")


    # to 3D and mirror
    exp3d = vector_to_3D(experimental, reciprocal_radius)
    exp3d_mirror = exp3d * np.array([-1,1,1])
    results =[]

    for sim_idx, trees in enumerate(precomputed_trees):
        best_score, best_rotation, mirror = float('inf'), 0.0, 1.0

        for rot_idx, sim_tree in enumerate(trees):
            # Points in current sim frame
            sim_points = sim_tree.data

            # total points 
            n_tot = len(exp3d) + len(sim_points)
            # Experimental tree
            exp_tree = cKDTree(exp3d)
            exp_tree_mirror = cKDTree(exp3d_mirror)

            # Original version
            dist_exp_to_sim, _ = sim_tree.query(exp3d,distance_upper_bound=distance_bound)
            dist_sim_to_exp, _ = exp_tree.query(sim_points,distance_upper_bound=distance_bound)


            n_unmatched_exp = np.sum(np.isinf(dist_exp_to_sim))
            n_unmatched_sim = np.sum(np.isinf(dist_sim_to_exp))
            matched_score = np.sum(dist_exp_to_sim[np.isfinite(dist_exp_to_sim)])
            # normalise the score
            score = (matched_score + unmatched_penalty * (n_unmatched_exp + n_unmatched_sim))/n_tot

            # Mirrored version
            dist_exp_to_sim_m, _ = sim_tree.query(exp3d_mirror,distance_upper_bound=distance_bound)
            dist_sim_to_exp_m, _ = exp_tree_mirror.query(sim_points,distance_upper_bound=distance_bound)

            n_unmatched_exp_m = np.sum(np.isinf(dist_exp_to_sim_m))
            n_unmatched_sim_m = np.sum(np.isinf(dist_sim_to_exp_m))
            matched_score_m = np.sum(dist_exp_to_sim_m[np.isfinite(dist_exp_to_sim_m)])
            # normalise score_mirror
            score_mirror = (matched_score_m + unmatched_penalty * (n_unmatched_exp_m + n_unmatched_sim_m))/n_tot


            if score < best_score:
                best_score = score
                ang = rot_idx * step_size_rad 
                best_rotation = wrap_degrees(ang) 
                mirror = 1.0

            if score_mirror < best_score:
                best_score = score_mirror
                ang = rot_idx * step_size_rad 
                best_rotation = wrap_degrees(ang) 
                mirror = -1.0
        results.append((sim_idx, best_score, best_rotation, mirror))
    # results = sorted(results, key = lambda x : x[1])[:n_best]
    results = sorted(results, key = lambda x : x[1])[:1]
    result_lst.append(np.array(results))
    # Return array of shape (len(experimental), n_best, 4)
    n_array = np.array(result_lst)
    return n_array
     

