import matplotlib.pyplot as plt
import numpy as np


"""
Fil for plottefunksjoner
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


