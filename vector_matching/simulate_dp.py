import numpy as np

def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


if __name__ == '__main__':
    # Parameters 
    image_size = 256
    sigma = 1.5
    kernel_radius = int(3*sigma)
    kernel_size = 2 * kernel_radius + 1
    center = image_size // 2

    # Load data
    IN_FILE = 'npy_files/sim_r_theta_intensity.npy'
    OUT_FILE = 'npy_files/150525_sim_dp_intensity_for_TM.npy'
    polar_data = np.load(IN_FILE, allow_pickle=True)
    num_patterns = polar_data.shape[0]

    # Estimate scaling factor
    max_r = np.max(polar_data[..., 0])
    scale_factor = (image_size * 0.4) / max_r

    # Gaussian kernel
    xg, yg = np.meshgrid(np.arange(kernel_size) - kernel_radius,
                         np.arange(kernel_size) - kernel_radius)

    gaussian_kernel = np.exp(-(xg**2 + yg**2) / (2 * sigma**2))
    gaussian_kernel /= gaussian_kernel.max()

    # main loop
    data_stack = np.zeros((num_patterns, image_size, image_size),dtype=np.float32)

    for i in range(num_patterns):
        r_theta_intensity = polar_data[i]
        r, theta = r_theta_intensity[:, 0], r_theta_intensity[:, 1]

        if r_theta_intensity.shape[1] == 3:
            intensity = r_theta_intensity[:, 2]
        else:
            intensity = np.ones_like(r)

        # remove padding
        mask = ~((r==0) & (theta == 0))
        r_masked = r[mask]
        theta_masked = theta[mask]

        # Convert from polar to cart
        x, y = pol2cart(r_masked, theta_masked)
        x_px = (x * scale_factor + center).astype(int)
        y_px = (y * scale_factor + center).astype(int)

        # Add Gaussians
        for xp, yp, amp in zip(x_px, y_px,intensity):
            xs = slice(xp - kernel_radius, xp + kernel_radius + 1)
            ys = slice(yp - kernel_radius, yp + kernel_radius + 1)
            if 0 <= xs.start and xs.stop <= image_size and 0 <= ys.start and ys.stop <= image_size:
                data_stack[i, ys, xs] += amp * gaussian_kernel

    np.save(OUT_FILE, arr=data_stack, allow_pickle=True)
            
