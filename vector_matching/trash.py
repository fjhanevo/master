# Turn on nitro boost
@njit()
def method1(exp_peaks, sim_peaks,cut_off):
    """
    This method tries sum(|exp - sim|) to find 
    the shift between the datasets
    exp_peaks : (N,2)
    sim_peaks : (M,N,2)
    The logic here is that we loop through all the simulated frames (M)
    and finds the abs value between r positions in both datasets
    to find the dTheta shift. 
    """
    
    # Keep track of which experimental frame we're at
    frame = -1
    # Store the score 
    score = []

    # Loop through all simulated frames
    for sim_peak in sim_peaks:
        # Reset this value at the beginning of each new itertion
        frame = 0
        # sum = exp_peaks[]
        


#NOTE: Denne fungerer bare for exp_peak.shape == sim_peak.shape!!
def get_min_distance_procrustes(exp_peak, sim_peaks):
    """
    Take in ONE experimental frame of shape (N,2),
    Compare it with all simulated frames of shape (N,2)
    Return the minimum distance
    """
    # Store disparites from algorithm here
    disparity = []

    # Loop through all the simulated frames (M,N,2)
    # to get the (N,2) frames so we can actually compare them
    for sim_peak in sim_peaks:
        _, _, disp = procrustes(exp_peak, sim_peak[...])
        
        # Append the disparity for each frame
        disparity.append(disp)
    return min(disparity)


    # This is how to get (N,2) shape of sim_peaks
    # for sim_peak in sim_peaks:
    #     print(sim_peak[...].shape)

    sim_frame = sim_peaks[frame]
    # prøver å legge til min_val i vektoren for å se om det gjør noe
    # prøv bare med r først, se hva som skjer
    good_frame[...][0] -= min_val
    good_frame[...][1] -= min_val

    plt.scatter(good_frame[...,1],good_frame[...,0],color='blue', label='exp')
    plt.scatter(sim_frame[...,1], sim_frame[...,0], color='red', label='sim')
    plt.xlabel(r'$\theta$')
    plt.ylabel('r')
    plt.legend()
    plt.show()#NOTE: hyperspy og pyxem suger, prøv å se om de er like på en annen måte

def load_and_check_match(filename, sim_frame, i):
    dp = hs.load(filename)
    # Denne fungerer ikke så ser vekk fra det nå
    # m = sim_frame.to_markers(sizes=5,color='red')
    dp = dp.inav[i:i+1]
    # funker ikke 
    hs.plot.plot_signals([dp,sim_frame], norm='log',cmap='viridis_r')
    plt.show()
# Nitro boost doesn't work for this one:(
#@njit()
#NOTE: Wow den fungerer!
def emd_match(exp_peak, sim_peaks):
    """
    Uses the Wasserstein-/Earth's mover distance(EMD) to find the
    similarity between the experimental and simulated frames
    exp_peak is of shape (N,2) where N varies (which is sad)
    and sim_peaks is of dimension (D,2) where D = 34 (which is happy)
    """
    
    # Store the score here
    disparities = []
    # Loop through sim_peaks to get the correct dimension
    # Use enmuerate to track which frame we're at
    t1 = time()
    for sim_peak in sim_peaks:
        disparities.append(wasserstein_distance_nd(exp_peak,sim_peak[...]))

    # Now that we have the disparities, we can find the one with the 
    # lowest value and link it to the corresponding simulated frame
    # Denne kan skrives om bedre....
    min_val = min(disparities)
    frame = -1
    for i in range(len(disparities)):
        if disparities[i] == min_val:
            frame = i
            break

    t2 = time()
    print(f"Computation time: {(t2-t1)/60} min")
    return min_val, frame 

# Henter tilbake Procrustes koden og tester den opp mot EMD
def procrustes_match(exp_peak, sim_peaks):
    """
    Take in ONE experimental frame of shape (N,2),
    Compare it with all simulated frames of shape (N,2)
    Return the minimum distance
    """
    # Store disparites from algorithm here
    print(exp_peak.shape)
    if exp_peak.shape != sim_peaks[0].shape:
        exp_peak=fix_shape(exp_peak,sim_peaks[0].shape)
    print(exp_peak.shape)
    disparities= []

    # Loop through all the simulated frames (M,N,2)
    # to get the (N,2) frames so we can actually compare them
    for sim_peak in sim_peaks:
        _, _, disp = procrustes(exp_peak, sim_peak[...])
        
        # Append the disparity for each frame
        disparities.append(disp)
    min_val = min(disparities)
    frame = -1
    for i in range(len(disparities)):
        if disparities[i] == min_val:
            frame = i
            break
    return min_val, frame

#TODO: Fix hvordan score blir lagt til, foreløpig funker det dårlig:/
def match_that_b(exp, sim):
    # loop through all exps
    score_exp = []
    score_sim=[]
    overlap = 7e-2
    for exp_pin in exp:
        for exp_points in exp_pin:
        # loop through all sims
            score_it = 0
            for s in sim:
                for sim_points in s:
                    score_s= 0
                    if np.abs(exp_points[0]-sim_points[0]) < overlap and np.abs(exp_points[1]-sim_points[1]) < overlap:
                        score_it +=1
                        score_s += 1
                    score_exp.append(score_it)
                    score_sim.append(score_s)

    frame_exp = 0
    frame_sim = 0
    exp_max = max(score_exp)
    sim_max = max(score_sim)

    for i in range(len(score_exp)):
        if score_exp[i] == exp_max:
            frame_exp = i
            break

    for i in range(len(score_sim)):
        if score_sim[i] == sim_max:
            frame_sim = i
            break

    return frame_exp, frame_sim

#### TING FRA main i playground!
strict3d = np.array(vector_to_3D(strict_peaks[exp_frame], reciprocal_radius))
strict_rot= full_z_rotation(strict3d,ang_step)
print(strict3d.shape)
print(strict_rot.shape)
# print("Original:")
# match_one_frame(exp3d,simulated,ang_step)
# print("Strict:")
# match_one_frame(strict3d, simulated, ang_step)

# sim263 = simulated[263]
# sim263 = full_z_rotation(sim263,ang_step)
# match_one_frame(sim263,simulated,ang_step)

# t1 = time()
# run_full_tree(org, simulated, reciprocal_radius, ang_step)
# t2 = time()
# print(f"Computation time: {(t2-t1)/60} min")


# lbls = ('exp[56]', 'sim[263]')
# plotting.plot_two_spheres(experimental[18])

### LA DETTE STÅ FOR MØTE ###
# frames, score, rotation = kdtree_match(experimental, simulated,ang_step)
# print('Best frames:', frames)
# print('Best score:', score)
# print('Rotation:', rotation)
# exp_org = np.array([apply_z_rotation(v,rotation) for v in experimental[0]])
# lbls = ('exp[56]','sim[263]')
# plotting.plot_two_spheres(exp_org,simulated[263],lbls)

# lbls = ('strict[56]','sim[688]')
# plotting.plot_two_spheres(strict_rot[49],simulated[688],lbls)

def count_overlap(exp_vector:np.ndarray, sim_vector:np.ndarray, tagged_points:list,overlap:float):
    """
    Takes in an experimental vector point and a simulated one.
    Loops through each entry to check for overlap. 
    Also check the list for tagged points to not tag a point twice.
    Determines overlap by taking the length of each vector and comparing it to
    the overlap value
    Should update tagged_points, so and maybe return it?
    """

    # Mulig det er en bedre måte å gjøre dette på....
    for exp_xy in exp_vector:
        # don't care about the z-component
        for sim_xy in sim_vector:
            # first check if exp_xy is in tagged_points
            if exp_xy.tolist() not in [points.tolist() for points in tagged_points]:
                if np.abs(exp_xy[0]-sim_xy[0]) < overlap and np.abs(exp_xy[1]-sim_xy[1]) < overlap:
                    # matching condition is met, add exp_xy to tagged_points
                    tagged_points.append(exp_xy)
                
def cool_sphere_match(exp_frame:np.ndarray, sim_frames:np.ndarray, step_size:float) -> tuple:
    """
    Temporary psuedocode for cool_sphere_match:
    Takes in an experimental frame (N,2) and all simulated frames (M,Q,2)
    Set a cut_off distance indicating when two points overlap
    For each simulated frame:
        Apply a rotation around the z-axis with a given step_size
        if overlap:
            mark overlapping exp point, making it not registraeble(?) for further overlap
            store rotation needed to get to this exp point from starting sim point
        Count how many points overlapped and append it to a list to keep track
    Find the max value of points in the score list 
    Find the frame that matches with this list
    return the best matching simulated frame
    """
    score = []
    tagged_points = []
    overlap = 7e-2  # Temporary overlap-distance
    # initialize space where to rotate over with a step size
    loop_vals = np.arange(0,2*np.pi,step_size).tolist()

    # loop through all sim_frames first
    #i = 0 # just to debug
    for sim_frame in sim_frames:
        # debug statement 
        # print(i)
        # Keep track of point overlap, reset after each iteration
        tagged_points.clear() 
        # now we're at the correct dimension for sim_frames
        # loop through the angular values and apply a rotation around the z-axis
        for ang_step in loop_vals:
            sim_frame = np.array([apply_z_rotation(frame,ang_step) for frame in sim_frame])
            # now we can check for overlap
            count_overlap(exp_frame, sim_frame, tagged_points, overlap)
        # track the score of each simulated frame

        # Make a log for debugging purposes
        # with open('out.txt','w') as f:
        #     with redirect_stdout(f):
        #         print("Frame:",sim_frame)
        #         print("Tagged points per frame:",len(tagged_points))

        score.append(len(tagged_points))
        # i+=1 # also sdebyg

    # return the simulated frame with the highest score
    frame = -1
    max_val = max(score)
    for i in range(len(score)):
        if score[i] == max_val:
            frame = i
    return frame, score


def test_cool_sphere_match(exp_frame:np.ndarray, sim_frames:np.ndarray, step_size:float) -> None:
    t1 = time()
    best_frame, score = cool_sphere_match(exp_frame,sim_frames,step_size)
    t2 = time()
    print(f"Computation time: {(t2-t1)/60} min")
    print('Best matching frame:', best_frame)
    labels = ('experimental', 'simulated')
    # Make a gif
    # make_rotating_sphere_gif(exp_frame,sim_frames[best_frame],labels)
    # then plot the two spheres
    # plotting.plot_two_spheres(exp_frame,sim_frames[best_frame], labels)
    # print('len score:',len(score))
    print('Max score:',max(score))
    print('Min score:',min(score))
    max_val = max(score)
    for i in range(len(score)):
        if i == max_val:
            print("Found one bitch")
    print("Score for frame:",best_frame)
    print(score[best_frame])

    """ Structure orix stuff"""
    # a=4.0495 # Lættis parameter
    # atoms = [Atom('Al', [0, 0, 0]), Atom('Al', [0.5, 0.5, 0]), Atom('Al', [0.5, 0, 0.5]), Atom('Al', [0, 0.5, 0.5])]
    # lattice = Lattice(a, a, a, 90, 90, 90)
    # phase = orix.crystal_map.Phase(name='Al', space_group=225, structure=Structure(atoms, lattice))
    #
    
    
    # Make simulated 3D, can do it this way cause its homogeneous
    # sim3d = np.array([sm.vector_to_3D(sim,reciprocal_radius) for sim in simulated])
    # exp3d = np.array(sm.vector_to_3D(experimental[exp_frame],reciprocal_radius))
    # exp3d = sm.full_z_rotation(exp3d,ang_step)
    # score, rotation = sm.get_rot_score_per_frame(exp3d,sim3d,ang_step) 

    # fig = plt.figure(figsize=(8,6))
    # ax = fig.add_subplot(111, projection='ipf', symmetry=phase.point_group)
    # ax.scatter(score, c=rotation, cmap='inferno')
    # plt.show()

# def full_kdtree_match(experimental, simulated, ang_step):
#     frames = []
#     scores = []
#     rotations=[]
#     score = float('inf')
#     best_frame = None
#     best_rot = None
#
#     for sim_idx, sim_frame in enumerate(simulated):
#         for exp_idx, exp_frame in enumerate(experimental):
#             tree = cKDTree(sim_frame)
#             distances, _ = tree.query(exp_frame)
#             score_i = np.sum(distances)
#
#             if score_i < score:
#                 score = score_i
#                 best_rot = -exp_idx*ang_step
#                 best_frame = sim_idx
#         frames.append(best_frame)
#         scores.append(score)
#         rotations.append(best_rot)
#     return frames, scores, rotations


def general_kdtree_match(experimental, simulated, ang_step, n_best, reciprocal_radius):
    """
    Match two datasets to determine which simulated frame fits best with each experimental 
    frame, where the simulated frames are fully rotated around the z-axis.
    The two datasets are transposed to 3D and the simulated frames are rotated around the 
    z-axis. 
    Tracks the frame, score and in-plane rotation for the frames, and determines the best 
    in-plane rotation for each frame. 

    Params:
        experimental: The 2D experimental dataset in polar coordinates
        simulated: The 2D simulated dataset in polar coordinates
        ang_step: The angular step at which the sphere is rotated
        n_best: How many scores to track, should return as (60, n_best, 4)
        reciprocal_radius: The reciprocal radius to evaluate for
    Returns:
        n_array: A dataset of shape (len(experimental), n_best, 4)
        where the last 4-array is the nx4 array defined as
        [index, correlation, in-plane, mirror_factor]
    """
    
    # Declare variables to store data
    frames = np.zeros(n_best, dtype=int)
    scores = np.full(n_best, float('inf'))
    rotations = np.zeros(n_best)
    results = []

    # Convert simulated to 3D
    sim3d = np.array([vector_to_3D(sim_vec, reciprocal_radius) for sim_vec in simulated])
    
    for sim_rot in sim3d:
        # Rotate the simulated frame
        sim_rot= full_z_rotation(sim_rot, ang_step)
        
        for sim_idx, sim_frame in enumerate(sim_rot):
            # Reset these after each iteration
            best_score = float('inf')
            best_rotation = 0.0
            best_frame = sim_idx
            for exp_frame in experimental:
                # Convert experimental to 3D
                exp3d = np.array([vector_to_3D(exp_frame, reciprocal_radius)])
                
                # KD-tree for fast nn-search
                tree = cKDTree(sim_frame)
                # find nn in sim_frame for each point in exp3d
                distances, _ = tree.query(exp3d)

                # calculates the score based on distances between points, low score is good
                score = np.sum(distances)

                # Track best match
                if score < best_score:
                    best_score = score
                    best_rotation = sim_idx * ang_step
                    best_frame = sim_idx
        # Somthing happens here that glues the function together
        ...
        lst = np.column_stack((frames, scores, rotations, np.ones_like(frames)))
        results.append(lst)
    n_array = np.array(results)
    # print(n_array.shape)
    return n_array


#NOTE: Har lyst til å teste denne om jeg har mer RAM
def fast_kdtree_sim_rot(experimental, simulated, ang_step):
    num_rotations = int(2 * np.pi / ang_step)

    # Precompute all sim rotations (only works if sim is homo)
    rot_sim = np.array([full_z_rotation(rot, ang_step) for rot in simulated])

    # Reshape for batch processing 
    rot_sim = rot_sim.reshape(-1, simulated.shape[1], 3)

    # Build kd-tree on exp data
    tree = cKDTree(experimental)

    # query all rotated versions at once
    distances, _ = tree.query(rot_sim)

    scores = distances.sum(axis=1)

    # Find best match (lowest score)

    best_frames = scores.reshape(len(simulated), num_rotations).argmin(axis=1)

    best_scores= scores.reshape(len(simulated), num_rotations).min(axis=1)
    best_rotations = best_frames * ang_step
    return np.arange(len(simulated)), best_scores, best_rotations

def slightly_fast_kdtree(experimental, simulated, ang_step):
    num_rotations = int(2 * np.pi / ang_step)

    best_matches = np.zeros(len(simulated), dtype=int)
    best_scores = np.full(len(simulated), float('inf'))
    best_rotations = np.zeros(len(simulated))

    # Big tree
    tree = cKDTree(experimental)

    for rot_idx in range(num_rotations):
        rot_sim = np.array([apply_z_rotation(sim, rot_idx * ang_step) for sim in simulated])

        # batch that b
        distances, _  = tree.query(rot_sim)

        # scoooooores (low socre is good)

        scores = np.sum(distances, axis=1)

        # Update if scores is lower
        mask = scores < best_scores
        best_scores[mask] = scores[mask]
        best_matches[mask] = np.arange(len(simulated))[mask]
        best_rotations[mask] = rot_idx * ang_step

    return best_matches, best_scores, best_rotations


def get_score_per_frame(experimental, simulated):
    """
    Now we only care about the score. 
    """

    tot_score = []
    # Loop through all points in experimental
    for sim_frame in simulated:
        # track score per frame
        score_per_frame = float('inf') 
        # loop through all simulated points 
        for exp_frame in experimental:

            #kdtree for fast nn-search
            tree = cKDTree(sim_frame)

            distances, _ = tree.query(exp_frame)

            score = np.sum(distances)

            if score < score_per_frame:
                score_per_frame = score
        tot_score.append(score_per_frame)
    return tot_score

def get_rot_score_per_frame(experimental, simulated,ang_step):
    """
    This tracks the score and the rotation at which the best score is for 
    one experimental frame when compared to all simulated frames.
    """
    score_tot = []
    rot_tot = [] 

    score = float('inf')
    rot = None
    # Loop through the simulated frames first 

    for sim_frame in simulated:
        # reset score after each sim_frame iteration 
        # loop through all experimental points
        for exp_idx, exp_frame in enumerate(experimental):
            # kdtree for nn search
            tree = cKDTree(sim_frame)
            distances,_ = tree.query(exp_frame)
            score_i = np.sum(distances)

            if score_i < score:
                score = score_i
                rot = exp_idx * ang_step
        score_tot.append(score) 
        rot_tot.append(rot)
    return np.array(score_tot), np.array(rot_tot)

#NOTE: Fjern, denne er obsolete, ettersom exp blir rotert

# def full_kdtree_match(experimental, simulated, ang_step):
#     
#     num_sim_frames = len(simulated)
#     frames = np.zeros(num_sim_frames,dtype=int)
#     scores = np.full(num_sim_frames, float('inf'))
#     rotations = np.zeros(num_sim_frames)
#
#     for sim_idx, sim_frame in enumerate(simulated):
#         # Reset these after each sim_frame iteration
#         best_score = float('inf')
#         best_rotation = 0
#         best_frame = sim_idx
#         for exp_idx, exp_frame in enumerate(experimental):
#             tree = cKDTree(sim_frame)
#             distances, _ = tree.query(exp_frame)
#             score = np.sum(distances)
#
#             # Check and update score criteria
#             if score < best_score:
#                 best_score = score
#                 best_rotation = exp_idx * ang_step
#                 best_frame = sim_idx
#
#         # Save frame, score and rotation after each iteration
#         frames[sim_idx] = best_frame
#         scores[sim_idx] = best_score
#         rotations[sim_idx] = best_rotation
#
#     return frames, scores, rotations
#

#NOTE: Samme som over

# def general_kdtree_match(experimental, simulated, ang_step, n_best):
#     num_sim_frames = len(simulated)
#     frames = np.zeros(num_sim_frames, dtype='int')
#     scores = np.full(num_sim_frames, float('inf'))
#     rotations = np.zeros(num_sim_frames)
#
#     for sim_idx, sim_frame in enumerate(simulated):
#         # Reset these bad boys after each iteration
#         best_score = float('inf')
#         best_rotation = 0
#         best_frame = sim_idx
#
#         for exp_idx, exp_frame in enumerate(experimental):
#             # kd-tree for fast nn search
#             tree = cKDTree(sim_frame)
#
#             # check nn-distance between points for the two datasets
#             distances, _ = tree.query(exp_frame)
#
#             # track score based on distances, low score is good
#             score = np.sum(distances)
#
#             # Check and update score criteria
#             if score < best_score:
#                 best_score = score
#                 best_rotation = exp_idx * ang_step
#                 best_frame = sim_idx
#         # Save frame, score and rotation after each iteration
#         frames[sim_idx] = best_frame
#         scores[sim_idx] = best_score
#         rotations[sim_idx] = best_rotation
#
        # Vet ikke helt hvordan jeg skal få den til å gi n_best

