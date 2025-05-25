import numpy as np
from numba import njit


def vector_to_3D(vector: np.ndarray,reciprocal_radius: float, dtype) -> np.ndarray:
    """
    Takes in a 2D polar vector and converts it to 
    a 3D sphere.
    Params:
    ---------
        vector: np.ndarray
            2D polar vector (r,theta)
        reciprocal_radius: float
            reciprocal space radius to account for
    Returns:
    ---------
        vector3d: np.ndarray
            3D vector coordinates
    """
    # get r coords
    r = vector[...,0]
    # get theta coords
    theta = vector[...,1]

    l = 2*np.arctan(r/(2*reciprocal_radius))

    # 3D coords
    x = np.sin(l)*np.cos(theta)
    y = np.sin(l)*np.sin(theta)
    z = np.cos(l)


    return np.stack((x,y,z), axis=-1).astype(dtype)

def _apply_z_rotation(vector: np.ndarray,theta: float, dtype) -> np.ndarray:
    """
    It just rotates the sphere around the z-axis with a given angle.
    Params:
    ---------
        vector: np.ndarray
            input 3D vector
        theta: float
            angle to rotate vector for in radians 
    Returns:
    ---------
        np.ndarray
            the dot product between rot matrix and input vector
    """

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot = np.array([[cos_theta, -sin_theta, 0],
                    [sin_theta, cos_theta, 0],
                    [0,0,1]], dtype=dtype)
    return vector @ rot.T

def _full_z_rotation(vector: np.ndarray, step_size: float, dtype) -> np.ndarray:
    """
    Helper function to add a rotation dimension.
    Params:
    ---------
        vector: np.ndarray
            3D input vector
        step_size: float
            angular increment
    Returns:
    ---------
        np.ndarray:
        fully rotated vector around the z-axis.
    """
    angles = np.arange(0,2*np.pi,step_size, dtype=dtype)
    rotated = [_apply_z_rotation(vector, theta, dtype) for theta in angles]
    return np.stack(rotated).astype(dtype)

def filter_sim(sim: np.ndarray, step_size: float, reciprocal_radius: float, dtype) -> np.ndarray:
    """
    Helper function for vector_match() to filter out zeros
    from sim because its homo, and now its in-homo.
    Params:
    ---------
        sim: np.ndarray
            simulated 3D vector
        step_size: float
            angular increment 
        reciprocal_radius: float
            reciprocal space radius to account for
    Returns:
    ---------
        full_z_rotation(): np.ndarray
            rotates the vector fully around the z-axis for given
            step_size, adds a new dimension in the dataset,
            from (N,3) to (M,N,3).
    """
    sim_filtered = sim[~np.all(sim == 0, axis=1)]

    sim_filtered_3d = _vector_to_3D(sim_filtered, reciprocal_radius, dtype)

    return _full_z_rotation(sim_filtered_3d,step_size, dtype)

@njit
def wrap_degrees(angle_rad: float, mirror: int) -> int:
    """
    Converts radian rotation into degrees, re-align reference axis
    with Pyxems conventions. Wraps it to the range (-180, 180) deg, 
    with counter-clockwise rotation
    Params:
    ---------
        angle_rad: float
            the in-plane angle found from VM in radians
        mirror: int
            mirror factor from VM, either 1 or -1 depending on if the 
            pattern was mirrored. Used to reverse rotation if mirrored
            (mirror == -1)
    Returns:
    ---------
        angle_deg: int 
            the in-plane rotation in degrees and changed to match Pyxems conventions.
            if mirrored, returns the reverse rotation
            if not mirrored, add 180 degs to match pyxem. 
    """
    angle_deg = int(np.rad2deg((angle_rad-np.pi/2)  % (2*np.pi)))
    if angle_deg > 180:
        angle_deg -= 360
    if mirror < 0:
        # reverse rotation when mirrored
        angle_deg = -angle_deg
        return angle_deg
    # Add 180 deg to match pyxem conventions (due to mirror factor applying a 180 deg rotation)
    else: 
        return angle_deg + 180


