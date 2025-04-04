import hyperspy.api as hs
import pyxem as pxm
import matplotlib.pyplot as plt
import numpy as np


def get_center_interactive(s):
    """
    Creates a circle to mask out the center beam in a diffraction pattern.
    Intended use is for a tilt series with a correctly centered beam at (0,0).
    Plots the entire tilt series in one frame for a quick estimation of the center.
    Change the radius/position of the circle interactively.
    Returns all coords from the CircleROI.
    To get only the radius do:
        _,_,radius,_ = get_center_interactive(s)
    This was the best way to do it afaik, cause hyperspy is not my friend;(
    """
    
    circ = hs.roi.CircleROI(cx=0,cy=0, r=1)
    s=s.max()
    s.plot(colorbar=None)
    circ.add_widget(s)
    plt.show()
    return circ 
    
def mask_center(s, radius, filename=None) -> None:
    """
    Masks the center beam of a tilt series with a given radius.
    Optionally, save the file
    """
    _, h,w = s.data.shape

    x = np.linspace(-1,1,h)
    y = np.linspace(-1,1,w)

    X,Y=np.meshgrid(x,y)

    # mask center
    mask = np.sqrt(X**2 + Y**2) > radius

    # apply mask to each frame
    masked_data = s.data * mask

    # set org data to masked data
    s.data = masked_data

    s.plot(cmap='viridis_r', norm='log' ,title='',colorbar=None)
    plt.show()
    if filename is not None:
        s.save(filename)

def log_shift(raw, base=10, shift=0.1):
    log_shift = np.log10(raw+shift) - np.log10(shift)
    return log_shift

def mask_peaks(s,min_dist=20,threshold=0.2,intensity=False,filename=None) -> None:
    """
    Mask the diffraction peaks after center beam is masked
    """
    st = s.template_match_disk(disk_r=2.2, subtract_min=False)
    vectors = st.get_diffraction_vectors(min_distance=min_dist, threshold_abs=threshold,get_intensity=intensity)
    vector_mask = vectors.to_mask(disk_r = 4)
    m = vectors.to_markers(sizes=5,color='red')

    s.plot(cmap='viridis_r',norm='log',title='',colorbar=False,
             scalebar=True,scalebar_color='black', axes_ticks='off')
    s.add_marker(m)
    plt.show()
    s_masked = s*vector_mask
    if filename is not None:
        s_masked.save(filename)

def peak_find_one_frame(dp,frame, **kwargs):
    st = dp.template_match_disk(disk_r=2.2, subtract_min=False)
    dp_i = dp.inav[frame:frame+1]
    st_i = st.inav[frame:frame+1]


    vectors = st_i.get_diffraction_vectors(**kwargs)
    # vm = vectors.to_mask(disk_r=4)
    m = vectors.to_markers(sizes = 5, color='red')

    print(vectors.data[0].shape)
    dp_i.plot(cmap='viridis_r',norm='log',title='',colorbar=False,
             scalebar=True,scalebar_color='black', axes_ticks='off')
    dp_i.add_marker(m)

    plt.show()
    return vectors.data

def get_peaks(dp, **kwargs):
    """
    Use if params has been specified to extract peaks from the entire dataset,
    gives an inhomogeneous dataset with shapes (N,2) where N indicates the number
    of peaks per frame.
    """
    st = dp.template_match_disk(disk_r = 2.2, subtract_min=False)
    vectors = st.get_diffraction_vectors(**kwargs)
    return vectors.data


if __name__ == '__main__':
    DIR_HSPY = 'processed_hspy_files/'
    DIR_NPY = 'npy_files/'
    FILE='LeftFish_unmasked.hspy'
    CENTER_FILE = 'LF_cal_log_m_center.hspy'
    SAVE_FILE = 'LF_cal_log_m_center_m_peaks.hspy'
    # VECTOR_FILE = 'LF_peaks_m_center_m_peaks.npy'
    VECTOR_FILE = 'peaks_all_LoG.npy'
    # dp = hs.load(DIR_HSPY+FILE)
    dp = hs.load(DIR_HSPY+CENTER_FILE)
    # _,_,radius,_= get_center_interactive(dp)
    # print(radius)
    # radius = 0.26425
    # mask_center(dp,radius,DIR_HSPY+CENTER_FILE)
    params_f29 = {
        'method': 'laplacian_of_gaussian',
        'get_intensity': False,
        'min_sigma': 4,
        'max_sigma': 5,
        'num_sigma': 1,
        'overlap': 0.1,
        'log_scale': True,
        'exclude_border': True
    }
    # file29 = 'f29_peaks.npy'
    # f29_data = peak_find_one_frame(dp,29,**params_f29)
    # np.save(file=DIR_NPY+file29, arr=f29_data, allow_pickle=True)
    # print("File saved:",DIR_NPY+file29)
    data = get_peaks(dp, **params_f29)
    np.save(file=DIR_NPY+VECTOR_FILE, arr=data, allow_pickle=True)
    print("File saved:",DIR_NPY+VECTOR_FILE)
   

    # params_f56= {
    #     'method': 'laplacian_of_gaussian',
    #     'get_intensity': False,
    #     'min_sigma': 4.,
    #     'max_sigma': 5,
    #     'num_sigma': 1,
    #     'overlap': 0.1,
    #     'log_scale': True,
    #     'exclude_border': True
    # }
    # file56 = 'f56_peaks.npy'
    # f56_data = peak_find_one_frame(dp,56,**params_f56)
    # np.save(file=DIR_NPY+file56, arr=f56_data, allow_pickle=True)
    # print("File saved:",DIR_NPY+file56)
