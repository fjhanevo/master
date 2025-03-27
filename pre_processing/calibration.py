import hyperspy.api as hs
import pyxem as pxm
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps

plt.rcParams.update({'font.size':18})
def get_trace(dp,i,x1,y1,x2,y2) -> None:

    img = dp.inav[i]

    img.plot(cmap='magma', norm='log',title='',colorbar=False)
    
    # Draw a line between DPs
    line = hs.roi.Line2DROI(x1=x1, y1=y1, x2=x2, y2=y2,linewidth=5)
    line2 = hs.roi.Line2DROI(x1=15, y1=52, x2=242, y2=204,linewidth=5)
    line.add_widget(img)
    line2.add_widget(img)
    plt.tight_layout()
    plt.show()

    print(line)

    # Trace img to find peak positions
    trace=line(img).as_signal1D(0)
    trace.plot(norm='log',title='')
    # trace=np.log(trace)
    xticks = np.arange(0,240,20)
    plt.xticks(xticks)
    plt.tight_layout()
    plt.show()
#
    # Find peaks
    print(sps.find_peaks(trace.data))

def plot_max(s):

    s2 = hs.load('UnderFish_calibrated.hspy')
    s2 = s2.sum()
    s = s.sum()
    # s = s.get_azimuthal_integral2d(npt=112)
    
    s = s2+s
    s.plot(cmap='magma_r', norm='log')
    # s2.plot(cmap='magma_r', norm='log')
    print(s.axes_manager)
    plt.show()
    

def set_calibration(dp, diffraction_calibration, name=None) -> None:
    dp.set_diffraction_calibration(diffraction_calibration)
    dp.plot(cmap='magma')
    if name is not None:
        dp.save(name)
    plt.show()

if __name__ == "__main__":
    # Load dataset
    dp = hs.load('UnderFish/SPED_FCC_Al_60_tilts.hspy')
    # FILE = 'UnderFish_interpolate_hsw30_subpixelTrue_sigma1_5_uf2_nearest.hspy'
    # dp = hs.load('UnderFish/'+FILE)
    # s = hs.load('LeftFish/LeftFish2_interpolate_hsw30_subpixelTrue_sigma1_5_uf2_nearest_inplaceFalse.hspy')
    # Al lattice parameter
    a=4.0495
    i = 37
    # x1, y1, x2, y2 = 189, 26, 62, 229
    x1, y1, x2, y2 = 189, 26, 62, 229
    # get_trace(dp,i,x1,y1,x2,y2)
    dist_111_peaks = a*(156-81)/2
    g_dist_112_axis = np.sqrt(8)/np.sqrt(3)
    diffraction_calibration = g_dist_112_axis/dist_111_peaks
    print(diffraction_calibration)
