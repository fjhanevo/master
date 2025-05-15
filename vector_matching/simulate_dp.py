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
    FILE = 'npy_files/LF_r_theta_sim.npy'
    polar_data = np.load(FILE, allow_pickle=True)
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
        r_theta = polar_data[i]
        r, theta = r_theta[:, 0], r_theta[:, 1]

        # remove padding
        mask = ~((r==0) & (theta == 0))
        r_masked = r[mask]
        theta_masked = theta[mask]

        # Convert from polar to cart
        x, y = pol2cart(r_masked, theta_masked)
        x_px = (x * scale_factor + center).astype(int)
        y_px = (y * scale_factor + center).astype(int)

        # Add Gaussians
        for xp, yp in zip(x_px, y_px):
            xs = slice(xp - kernel_radius, xp + kernel_radius + 1)
            ys = slice(yp - kernel_radius, yp + kernel_radius + 1)
            if 0 <= xs.start and xs.stop <= image_size and 0 <= ys.start and ys.stop <= image_size:
                data_stack[i, ys, xs] += gaussian_kernel

    np.save('npy_files/150525_simulated_dp_for_TM.npy', arr=data_stack, allow_pickle=True)
            
