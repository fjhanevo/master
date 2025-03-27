from time import time 
from numba import njit
import pyxem as pxm
import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt

"""
Fil relatert til ulike vector algortitmer
"""

def fix_shape(exp_peak, Q):
    """
    Så greia med denne funksjonen er at den tar inn exp_peak som er en (N,2) ndarray
    og den tar inn Q som er (Q,2) greia fra sim_peak, skjønner?
    Så forsikrer funksjonen oss om at exp_peaks matcher dimensionen til 
    sim_peaks slik at vi kan bruke de kule algoritmene til å matche datasettene.
    
    Hvis exp_peak er mindre enn Q fyller jeg bare med 0er, og hvis den 
    er større fjerner jeg det siste elementet bare for å teste om dette faktisk gir OK verdier
    eller om det er bare tull
    """
    while exp_peak.shape != Q:
        if (exp_peak.shape < Q):
            exp_peak=np.append(exp_peak,[[0,0,0]],axis=0)
        #TODO: Fiks det her når vi har N>Q, driter i det for nå
        # else:
        #     exp_peak.delete[-1]
    return exp_peak
    

def kabsch_match(exp_peak, sim_peaks):
    """
    Kabsch implementation. 
    """

    # Fix the shape first
    print(exp_peak.shape)
    if exp_peak.shape != sim_peaks[0].shape:
        exp_peak=fix_shape(exp_peak, sim_peaks[0].shape)
    print(exp_peak.shape)
    rmsd_vals= []

    # compute centroids
    # only need this once since its the same
    centroid_exp = np.mean(exp_peak, axis=0)

    rot_val = []
    # Loop through the fellas and do the kabsch match
    for sim_peak in sim_peaks:
        # compute sim centroid
        centroid_sim = np.mean(sim_peak,axis=0)

        # find optimal translation
        t = centroid_sim - centroid_exp

        # center the points
        exp = exp_peak - centroid_exp
        sim = sim_peak - centroid_sim

        # compute covariance matrix
        H = np.dot(exp.T, sim)

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # validate right-hand coordinate system
        if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
            Vt[-1, :] *= -1.0

        # optimal rotaiton
        R = np.dot(Vt.T, U.T)
        rot_val.append(R)

        # get root-mean-square-deviation
        rmsd = np.sqrt(np.sum(np.square(np.dot(exp,R.T) - sim)) / exp_peak.shape[0])

        # save values at each iteration
        rmsd_vals.append(rmsd)
    
    # find corresponding frame with min_val
    rmsd_frame = -1
    min_rmsd = min(rmsd_vals)
    for i in range(len(rmsd_vals)):
        if rmsd_vals[i] == min_rmsd:
            rmsd_frame= i
            break

    return min_rmsd, rmsd_frame, rot_val[rmsd_frame]

#NOTE: IKKe fornøyd med denne;(
def cool_sphere_match(exp_sphere,sim_sphere):
    """
    exp sphere (N,3)
    sim_sphere (M,Q,3)
    compare al exp points with sim points to get  
    a cool sphere
    score is set a sum(exp-sim), low score is good
    """

    score = []
    
    # Loop through all frames
    for sim in sim_sphere:
        # loop through all vectors
        temp = 0    # reset this bad boy after each iteration
        for exp in exp_sphere:
            for s in sim:
                temp += np.abs(exp-s)/sim.size
        # append after each frame iteration
        score.append(np.sum(temp))
             
    min_score = min(score)
    frame = -1
    # find corresponding frame to min_val 
    for i in range(len(score)):
        if score[i] == min_score:
            frame = i
            print("yes bitch")
    return min_score, frame

def test_matching_algorithms(exp_peak, sim_peaks):
     
    # min_val, frame = procrustes_match(exp_peak, sim_peaks)
    # print("Procrustes:")
    # print("The min value was found to be:", min_val)
    # print("With corresponding simulated frame:", frame)
    # print("------------------------------")
       # min_rot, rot_frame, min_rmsd, rmsd_frame = kabsch_match(exp_peak,sim_peaks) 
    min_rmsd, rmsd_frame,rot_val = kabsch_match(exp_peak,sim_peaks) 
    print("Kabsch:")
    # print("Optimal rotation:", min_rot)
    # print("Optimal rotation frame:", rot_frame)
    print("RMSD:", min_rmsd)
    print("RMSD frame:", rmsd_frame)
    print("Rotation:",rot_val)
    print("------------------------------")
    # min_val, frame = cool_sphere_match(exp_peak, sim_peaks)
    # print("Cool sphere match:")
    # print("Min_val:", min_val)
    # print("Frame:", frame)

def load_and_check_match(filename, sim_frame, i):
    dp = hs.load(filename)
    sim_frame = pxm.signals.DiffractionVectors2D(sim_frame)
    m = sim_frame.to_markers(sizes=5,color='red')
    dp = dp.inav[i:i+1]
    dp.plot(cmap='viridis_r', scalebar_color='black',colorbar=None)
    dp.add_marker(m)
    plt.show()

def vector_to_3D(vector,reciprocal_radius):
    """
    Takes in a 2D polar vector and converts it to 
    a 3D dataset.
    """
    R = reciprocal_radius
    # get r coords
    r = vector[...,0]
    # get theta coords
    theta = vector[...,1]

    l = 2*np.arctan(r/(2*R))

    # 3D coords
    x = np.sin(l)*np.cos(theta)
    y = np.sin(l)*np.sin(theta)
    z = np.cos(l)

    # create new dataset
    vector3d = np.stack([x,y,z],axis=-1)

    return vector3d

def plot3D(vec):
    ax=plt.axes(projection='3d')
    ax.scatter3D(vec[...,0],vec[...,1],vec[...,2],cmap='inferno')
    plt.show()

def plot_exp_sim3D(vec1,vec2):
    ax=plt.axes(projection='3d')
    ax.scatter3D(vec1[...,0],vec1[...,1],vec1[...,2],label="exp")
    ax.scatter3D(vec2[...,0],vec2[...,1],vec2[...,2],label="sim")
    plt.legend()
    plt.show()

if __name__ == '__main__': 
    DIR_HSPY = 'processed_hspy_files/'
    DIR_NPY = 'npy_files/'
    FILE_EXP_L = 'LF_peaks_masked_center.npy'
    # FILE_EXP_U = 'UnderPeaks_xy.npy'
    FILE_SIM = 'r_theta_sim.npy'
    FILE_ORG = 'LF_cal_log_center_masked.hspy'
    FILE_BEAM = 'LeftFish_unmasked.hspy'

    # Load experimental peaks
    exp_peaks = np.load(DIR_NPY+FILE_EXP_L, allow_pickle=True)
    # Convert to polar
    exp_peaks = pxm.signals.DiffractionVectors(exp_peaks)
    exp_peaks = exp_peaks.to_polar()
    # Back to ndarray
    exp_peaks = exp_peaks.data
    # Load simulated peaks 
    sim_peaks = np.load(DIR_NPY+FILE_SIM, allow_pickle=True)
    reciprocal_radius = 1.35 
    
    # dist_cutoff = 7e-2 # Subject to change
    

    p563D = vector_to_3D(exp_peaks[56],reciprocal_radius)
    sim_sphere = [vector_to_3D(sim_peak,reciprocal_radius) for sim_peak in sim_peaks]
    sim = np.array(sim_sphere)
    # cool_sphere_match(p563D,sim)
    # print(p563D[0])

    # check matching algorithms for spheres
    # test_matching_algorithms(p563D,sim)
    rmsd_val, opt_frame, opt_rot = kabsch_match(p563D,sim) 
    print(opt_rot.shape)
    print(opt_rot)
    # get the rotation 
    r = R.from_euler('zxz',opt_rot,degrees=False)
    # apply rotation to sim

    sim_rotated = [r.apply(sim_rot) for sim_rot in sim[opt_frame]]

    sim_rot=np.array(sim_rotated)
    print(sim_rot.shape)
    plot_exp_sim3D(p563D,sim[770])
    
    # frame 770
    # plot_exp_sim3D(p563D,sim[3745])
    print(sim[3745])
    # plot3D(sim[3745])

