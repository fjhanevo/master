import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
params = {
    'figure.figsize':(8.0,4.0), 'axes.grid': True,
    'lines.markersize': 8, 'lines.linewidth': 2,
    'font.size': 18
}
plt.rcParams.update(params)



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

    plt.tight_layout()
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

def mask_background(data, filename:str, **kwargs) -> None:
    st = data.template_match_disk(disk_r = 2.2, subtract_min=False)
    vectors = st.get_diffraction_vectors(**kwargs)
    vector_mask = vectors.to_mask(disk_r = 4)
    data_masked = data*vector_mask
    data_masked.save(filename)

if __name__ == '__main__':
    DIR_HSPY = 'processed_hspy_files/'
    DIR_NPY = 'npy_files/'
    FILE='LeftFish_unmasked.hspy'
    CENTER_FILE = 'LF_cal_log_m_center.hspy'
    SAVE_FILE = 'LF_cal_log_m_center_m_peaks.hspy'
    FILE_STRICT = 'LF_cal_log_m_center_strict_peaks.hspy'
    # VECTOR_FILE = 'LF_peaks_m_center_m_peaks.npy'
    VECTOR_FILE = 'peaks_all_LoG.npy'
    # dp = hs.load(DIR_HSPY+FILE)
    dp = hs.load(DIR_HSPY+CENTER_FILE)
    # _,_,radius,_= get_center_interactive(dp)
    # print(radius)
    # radius = 0.26425
    # mask_center(dp,radius,DIR_HSPY+CENTER_FILE)
    params = {
        'method': 'laplacian_of_gaussian',
        'get_intensity': False,
        'min_sigma': 4,
        'max_sigma': 5,
        'num_sigma': 1,
        'overlap': 0.1,
        'log_scale': True,
        'exclude_border': True
    }
    params_intensity = {
        'method': 'laplacian_of_gaussian',
        'get_intensity': True,
        'min_sigma': 4,
        'max_sigma': 5,
        'num_sigma': 1,
        'overlap': 0.1,
        'log_scale': True,
        'exclude_border': True
    }
    params_liberal = {
        'method': 'laplacian_of_gaussian',
        'get_intensity': False,
        'min_sigma': 2.8,
        'max_sigma': 5,
        'num_sigma': 4,
        'overlap': 0.1 ,
        'log_scale': True,
        'exclude_border': True

    }
    # filename = 'peaks_intensity_all_LoG.npy'
    filename = '310525_liberal_peaks_for_discussion_LoG.npy'
    # peaks = get_peaks(dp,**params_intensity)
    # np.save(file=DIR_NPY+filename, arr=peaks,allow_pickle=True)
    f56 = peak_find_one_frame(dp, 29, **params_liberal)
    f56 = peak_find_one_frame(dp, 56, **params_liberal)
    # print(f56.shape)
    # f29 = peak_find_one_frame(dp, 29, **params)
    # dp_masked = hs.load(DIR_HSPY+FILE_STRICT)
    # dp_masked_56 = dp_masked.inav[56]
    # dp_masked_56.plot(cmap='viridis_r',norm='log',title='',colorbar=False,
    #          scalebar=True,scalebar_color='black', axes_ticks='off')
    #
    # plt.tight_layout()
    # plt.show()
    # dp_masked_29 = dp_masked.inav[29]
    # dp_masked_29.plot(cmap='viridis_r',norm='log',title='',colorbar=False,
    #          scalebar=True,scalebar_color='black', axes_ticks='off')
    # plt.tight_layout()
    # plt.show()
    #
    peaks_liberal = get_peaks(dp, **params)
    np.save(file=DIR_NPY+filename, arr=peaks_liberal, allow_pickle=True)

