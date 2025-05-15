import hyperspy.api as hs
from orix.plot import IPFColorKeyTSL
from orix.vector import Vector3d
import matplotlib.pyplot as plt
import numpy as np

#NOTE: Change figsize for ipf!
params = {
    'figure.figsize':(8.0,4.0), 'axes.grid': True,
    'lines.markersize': 8, 'lines.linewidth': 2,
    'font.size': 18
}
plt.rcParams.update(params)

"""
File for plotting everything.
"""

def plot_two_spheres(s1:np.ndarray,s2:np.ndarray,labels:tuple) -> None:
    l1, l2 = labels 
    ax = plt.axes(projection='3d')
    ax.scatter3D(s1[...,0], s1[...,1], s1[...,2], label=l1)
    ax.scatter3D(s2[...,0], s2[...,1], s2[...,2], label=l2)
    ax.grid(visible=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Set limites for axes
    # ax.set_xlim([-1,1])
    # ax.set_ylim([-1,1])
    # ax.set_zlim([-1,1])
    plt.legend()
    plt.show()

def plot_spheres_to_gif(s1:np.ndarray,s2:np.ndarray,labels:tuple) -> None:
    l1, l2, filename = labels

    ax = plt.axes(projection='3d')
    ax.scatter3D(s1[...,0], s1[...,1], s1[...,2], label=l1)
    ax.scatter3D(s2[...,0], s2[...,1], s2[...,2], label=l2)
    ax.grid(visible=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Set limits for axes (subject to change)
    ax.set_xlim([-0.9,0.9])
    ax.set_ylim([-0.7,0.7])
    ax.set_zlim([0.4,1])

    plt.legend()
    plt.savefig('vector_matching/sphere_gif/' + filename)
                 
def plot_spheres_with_axis_lims(s1:np.ndarray, s2:np.ndarray,labels:tuple) -> None:
    l1, l2,  = labels
    ax = plt.axes(projection='3d')
    ax.scatter3D(s1[...,0], s1[...,1], s1[...,2], label=l1)
    ax.scatter3D(s2[...,0], s2[...,1], s2[...,2], label=l2)

    ax.grid(visible=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    ax.set_xlim([-0.9,0.9])
    ax.set_ylim([-0.7,0.7])
    ax.set_zlim([0.4,1])
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_save_spheres(s1:np.ndarray, s2:np.ndarray,labels:tuple) -> None:
    """
    Plots two spheres and saves them to folder
    """
    l1, l2,filename  = labels
    ax = plt.axes(projection='3d')
    ax.view_init(elev=-90,azim=90,roll=0)
    ax.scatter3D(s1[...,0], s1[...,1], s1[...,2], label=l1)
    ax.scatter3D(s2[...,0], s2[...,1], s2[...,2], label=l2)

    ax.grid(visible=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    ax.set_xlim([-0.9,0.9])
    ax.set_ylim([-0.7,0.7])
    ax.set_zlim([0.4,1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)


def plot_in_plane(data:np.ndarray) -> None:
    plt.figure(figsize=(8,6))

    for i in range(data.shape[0]):
        plt.scatter(i, data[i])

    plt.ylabel('in-plane [rad]')
    plt.xlabel('Frame')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_score_per_frame(data:list, labels:tuple):
    plt.figure(figsize=(10,8))

    min_val = min(data)
    for i in range(len(data)):
        plt.scatter(i,data[i], c='black')
        if data[i] == min_val:
            plt.scatter(i,data[i], c='red', 
                        marker = 'o',
                        s=54*4,
                        label=f'frame: {i}, min_val: {data[i]}')
        
        
    lx, ly = labels
    plt.xlabel(lx)
    plt.ylabel(ly)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_2D_plane(s1:np.ndarray, s2:np.ndarray, labels:tuple) -> None:
    l1, l2 = labels
    ax = plt.axes(projection='3d')
    ax.view_init(elev=-90,azim=90,roll=0)
    ax.scatter3D(s1[...,0], s1[...,1], label=l1)
    ax.scatter3D(s2[...,0], s2[...,1], label=l2)
    ax.grid(visible=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.legend()
    plt.show()

def plot_2D_plane_save(s1:np.ndarray, s2:np.ndarray, labels:tuple) -> None:
    l1, l2, filename = labels
    ax = plt.axes(projection='3d')
    ax.view_init(elev=-90,azim=90,roll=0)
    ax.scatter3D(s1[...,0], s1[...,1], label=l1)
    ax.scatter3D(s2[...,0], s2[...,1], label=l2)
    ax.grid(visible=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-3.5,3.5])
    ax.set_ylim([-3.5,3.5])
    # ax.set_zlim([0.4,1])

    plt.legend()
    plt.savefig('vector_matching/sphere_gif/'+filename)


### Orientation Mapping plots ###
def plot_ipf(data, idx, phase, orientation ,cmap:str):
    """
    Plots an IPF with a red marker indicating the best found orientation.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='ipf', symmetry=phase.point_group)

    correlations = data.inav[idx].data[:,1]
    tm_indices = (data.inav[idx].data[:,0]).astype('int16')
    orientations = orientation[tm_indices]
    loris= data.to_single_phase_orientations()
    loris_best = loris[idx,0]
    ax.scatter(orientations, c=correlations, cmap=cmap)
    ax.scatter(loris_best,c='red',marker='o',s=100) # best found orientation
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

def plot_misorientation_hist(data):
    """
    Plots the misorientations as a histogram.
    """
    loris = data.to_single_phase_orientations()
    loris_best = loris[:,0]
    loris_ang  = loris_best.angle_with_outer(loris_best, degrees=True)
    plt.figure()
    plt.hist(loris_ang.flatten(), bins=60)
    plt.xlabel('Degrees')
    plt.ylabel('Count')
    plt.show()

def plot_misorientation_scatter(data):
    """
    Plots the misorienations as scatter plot.
    """
    loris = data.to_single_phase_orientations()
    loris_best = loris[:,0]
    loris_ang = loris_best.angle_with_outer(loris_best,degrees=True)

    plt.figure(figsize=(8,6))
    for i in range(len(loris_ang)-1):
        plt.scatter(i, loris_ang[i,i+1], s=34,c='black')

    plt.axhline(y = 1, color='red', label=r'1$\degree$', linestyle='dashed')
    plt.grid(True)
    plt.ylabel(r'Misorientation$\degree$',fontsize='26')
    plt.xlabel('Tilt Step',fontsize='26')
    plt.xticks(fontsize='18')
    plt.yticks(fontsize='18')
    plt.legend(fontsize='18', loc='center left')
    plt.tight_layout()
    plt.show()

def plot_crystal_map(results,phase):
    """
    Assumes reshaped dataset!
    """
    xmap = results.to_crystal_map()
    oris = xmap.orientations
    corrs = results.data[:,:,0,1].flatten()
    print(corrs.shape)

    key_x = IPFColorKeyTSL(phase.point_group, Vector3d.xvector())
    key_y = IPFColorKeyTSL(phase.point_group, Vector3d.yvector())
    key_z = IPFColorKeyTSL(phase.point_group, Vector3d.zvector())

    oris_z = key_z.orientation2color(oris)[:,:]
    xmap.plot(oris_z, overlay=corrs, remove_padding=True)
    plt.show()
    oris_x = key_x.orientation2color(oris)[:,:]
    xmap.plot(oris_x, overlay=corrs, remove_padding=True)
    plt.show()
    oris_y = key_y.orientation2color(oris)[:,:]
    xmap.plot(oris_y, overlay=corrs, remove_padding=True)
    plt.show()


def plot_compare_misorientation_scatter(data1, data2, data3, lbls:tuple, clrs:tuple):
    """
    Plots the misorientations as scatter plots for multiple datasets.
    """
    datasets = [data1, data2, data3]
    c1, c2, c3 = clrs
    colors = [c1,c2,c3]
    l1, l2, l3 = lbls 
    labels = [l1, l2, l3] 

    plt.figure(figsize=(8,6))

    for data, color, label in zip(datasets, colors, labels):
        loris = data.to_single_phase_orientations()
        loris_best = loris[:, 0]
        loris_ang = loris_best.angle_with_outer(loris_best, degrees=True)

        for i in range(len(loris_ang) - 1):
            plt.scatter(i, loris_ang[i, i + 1], s=34, c=color, label=label if i == 0 else "")

    plt.axhline(y=1, color='red', label=r'1$\degree$', linestyle='dashed')
    plt.grid(True)
    plt.ylabel(r'Misorientation$\degree$', fontsize=26)
    plt.xlabel('Tilt Step', fontsize=26)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18, loc='center left')
    plt.tight_layout()
    plt.show()

def plot_ipf_all_best_orientations(data, phase , cmap:str) -> None:


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='ipf', symmetry=phase.point_group)

    loris = data.to_single_phase_orientations()
    num_frames = data.axes_manager.navigation_shape[0]
    for idx in range(num_frames):
        correlations = data.inav[idx].data[:,1]
        loris_best = loris[idx, 0]
        ax.scatter(loris_best, c=correlations, cmap=cmap)

    plt.show()


