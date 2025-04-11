import numpy as np
from scipy.spatial import cKDTree
from sphere_matching import vector_to_3D, filter_sim
from time import time

def vector_match_score_test(
        experimental, 
        simulated, 
        step_size, 
        reciprocal_radius,
        distance_bound=0.05,
        unmatched_penalty=0.75
):
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

            # Experimental tree
            exp_tree = cKDTree(exp3d)
            exp_tree_mirror = cKDTree(exp3d_mirror)

            # Original version
            dist_exp_to_sim, _ = sim_tree.query(exp3d,distance_upper_bound=distance_bound)
            dist_sim_to_exp, _ = exp_tree.query(sim_points,distance_upper_bound=distance_bound)

            n_unmatched_exp = np.sum(np.isinf(dist_exp_to_sim))
            n_unmatched_sim = np.sum(np.isinf(dist_sim_to_exp))
            matched_score = np.sum(dist_exp_to_sim[np.isfinite(dist_exp_to_sim)])
            score = matched_score + unmatched_penalty * (n_unmatched_exp + n_unmatched_sim)

            # Mirrored version
            dist_exp_to_sim_m, _ = sim_tree.query(exp3d_mirror,distance_upper_bound=distance_bound)
            dist_sim_to_exp_m, _ = exp_tree_mirror.query(sim_points,distance_upper_bound=distance_bound)

            n_unmatched_exp_m = np.sum(np.isinf(dist_exp_to_sim_m))
            n_unmatched_sim_m = np.sum(np.isinf(dist_sim_to_exp_m))
            matched_score_m = np.sum(dist_exp_to_sim_m[np.isfinite(dist_exp_to_sim_m)])
            score_mirror = matched_score_m + unmatched_penalty * (n_unmatched_exp_m + n_unmatched_sim_m)


            if score < best_score:
                best_score = score
                best_rotation = np.rad2deg(rot_idx * step_size_rad)
                mirror = 1.0

            if score_mirror < best_score:
                best_score = score_mirror
                best_rotation = np.rad2deg(rot_idx * step_size_rad)
                mirror = -1.0
        results.append((sim_idx, best_score, best_rotation, mirror))
    # results = sorted(results, key = lambda x : x[1])[:n_best]
    results = sorted(results, key = lambda x : x[1])[:1]
    result_lst.append(np.array(results))
    # Return array of shape (len(experimental), n_best, 4)
    n_array = np.array(result_lst)
    return n_array

def test_params_and_save(exp, exp_frame,sim, rot, reciprocal_radius, dist_bound, penalty):
    exp = exp[exp_frame]
    # n_array = sm.vm_one_frame_take_two(exp,sim,ang_step,reciprocal_radius, n_best_candidates=n_best)
    n_array = vector_match_score_test(
        exp,sim,rot,reciprocal_radius,
        distance_bound=dist_bound, unmatched_penalty=penalty
    )
    n_best = n_array[0][0]
    frame, score, rotation, mirror = n_best[0],n_best[1],n_best[2], n_best[3]

    exp3d = sm.vector_to_3D(exp,reciprocal_radius)
    rot = np.deg2rad(rot)
    if mirror < 0.0:
        exp3d *= np.array([-1,1,1])
    sim = sim[int(frame)]
    sim_filtered = sim[~np.all(sim==0,axis=1)]
    sim_filtered3d = sm.vector_to_3D(sim_filtered, reciprocal_radius)
    sim_filtered3d_rot = np.array([sm.apply_z_rotation(vec,rot) for vec in sim_filtered3d])
    sim_str = 'sim['+str(int(frame))+']'
    exp_str = 'exp['+str(exp_frame)+']'
    dir = 'vector_matching/tmp_plots/'
    filename = dir+'f'+str(exp_frame)+'_dist'+str(dist_bound)+'_pen'+str(penalty)+'.png'
    lbls = (sim_str, exp_str, filename)
    
    plotting.plot_save_spheres(sim_filtered3d_rot, exp3d, lbls)


