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


def get_vectors(s,min_dist=20,threshold=0.2,intensity=False,filename=None) -> None:
    """
    Finds the diffraction peaks of a tilt series. 
    min_dist and threshold are set to the default values after trial and error,hihi.
    Optionally: Include the intensity, although it gives some weird values, 
    and will give a (N,3) ndarray instead of a (N,2) ndarray, where N 
    refers to the number of peaks found in each frame. 
    Also save the file to a npy-file if you want to.
    """
    st = s.template_match_disk(disk_r=2.2, subtract_min=False)
    vectors = st.get_diffraction_vectors(min_distance=min_dist,threshold_abs=threshold,
                                         get_intensity=intensity)
    if filename is not None:
        np.save(file=filename,arr=vectors.data,allow_pickle=True)

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


if __name__ == '__main__':
    DIR_HSPY = 'processed_hspy_files/'
    DIR_NPY = 'npy_files/'
    FILE='LeftFish_unmasked.hspy'
    CENTER_FILE = 'LF_cal_log_m_center.hspy'
    SAVE_FILE = 'LF_cal_log_m_center_m_peaks.hspy'
    # VECTOR_FILE = 'LF_peaks_m_center_m_peaks.npy'
    VECTOR_FILE = 'LF_peaks_masked_center.npy'
    # dp = hs.load(DIR_HSPY+FILE)
    dp = hs.load(DIR_HSPY+CENTER_FILE)
    # _,_,radius,_= get_center_interactive(dp)
    # print(radius)
    # radius = 0.26425
    # mask_center(dp,radius,DIR_HSPY+CENTER_FILE)

    ###### Test out some vals for min_distance and threshold ########
    # dist = 20
    # threshold = 0.5
    # get_vectors(dp, min_dist=dist,threshold=threshold, filename=DIR_NPY+VECTOR_FILE)
    # mask_peaks(dp,min_dist=dist,threshold=threshold, filename=DIR_HSPY+SAVE_FILE)
    # mask_peaks(dp)
    st = dp.template_match_disk(disk_r=2.2, subtract_min=False)
    vectors = st.get_diffraction_vectors(method='laplacian_of_gaussian',
                                         get_intensity=False,
                                         min_sigma=4.5,
                                         max_sigma=50,
                                         num_sigma=10,
                                         overlap=0.1,
                                         log_scale=False,
                                         exclude_border=True)

    # np.save(file=DIR_NPY+'LF_strict_peaks_log.npy',arr=vectors.data,allow_pickle=True)
    # print(vectors.data.shape)
    # print(vectors[56].data.shape)
    vm =  vectors.to_mask(disk_r=4)
    m =vectors.to_markers(sizes=5, color='red')

    dp.plot(cmap='viridis_r',norm='log',title='',colorbar=False,
             scalebar=True,scalebar_color='black', axes_ticks='off')
    dp.add_marker(m)
    plt.show()
    # mask_peaks(dp)
    # plot to see modified dataset
    # s = hs.load(DIR_HSPY+SAVE_FILE)
    # s.plot(cmap='viridis_r', norm='log', colorbar=False, scalebar_color='black')
    # plt.show()

