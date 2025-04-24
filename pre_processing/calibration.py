import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps

plt.rcParams.update({'font.size':18})

def get_trace(dp, frame, coords:tuple):

    x1,y1,x2,y2 = coords

    dp_frame = dp.inav[frame]

    dp_frame.plot(cmap='viridis_r', norm='log', title='', colorbar=False)

    line = hs.roi.Line2DROI(x1=x1,y1=y1,x2=x2,y2=y2, linewidth=5)
    line.add_widget(dp_frame, color='red')
    plt.tight_layout()
    plt.show()

    print(line)
    # trace to get pixel peak distances
    trace = line(dp_frame).as_signal1D(0)
    trace.plot(norm='log',title='')
    # xticks = np.arange(0,340,20)
    plt.title('')
    # plt.xticks(xticks)
    plt.tight_layout()
    plt.show()

    print(sps.find_peaks(trace.data))

if __name__ == "__main__":
    DIR_HSPY = 'processed_hspy_files/'
    FILE = 'LeftFish_to_calibrate.hspy'
    # Load dataset
    dp = hs.load(DIR_HSPY+FILE)
    a = 4.0495  # Al lattice parameter
    frame = 56
    coords = (11,230,249,22)
    get_trace(dp,frame, coords)
    dist_A = (161-46)/2
    dist_B = (156-56)/2
    g_dist = 2.0/np.sqrt(3)
    print(g_dist/(dist_B+dist_A))



