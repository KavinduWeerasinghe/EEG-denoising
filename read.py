import numpy as np
import wfdb
from scipy import signal
import mne
import matplotlib.pyplot as plt
import pywt
from skimage.restoration import denoise_wavelet

def mov_avg_filt(y,win=4):
    ret = np.cumsum(y, dtype=float)
    ret[win:] = ret[win:] - ret[:-win]
    ret=ret[win-2:] / win
    ret= np.pad(ret, (0,2), 'constant')
    return ret

def bandpass_filt(Y):
    sos = signal.cheby1(2, 1, Wn=[0.5,40], btype='bandpass', fs=512, output='sos')
    filtered = signal.sosfilt(sos, Y)
    return filtered

def show_wave(r,samples,):
    axs.plot(samples,r)

def open_record(rec_num : str)->tuple:
    """This function opens a given recording in NST database.

    :param rec_num: The number/name combination of NST databse
    :type rec_num: str
    
    :return: Readings of the NST record
    :rtype: tuple(object,np.ndarray,np.ndarray)
        :param raw_record: contains an object with several parameters as attributes
        :param ecg_signal: contains an array containing ECG signal values
        :param ecg_samples: contains a sample array (x values) in response to ecg_signal
        
    database location : D:/Intership/Template matching/databases/data_NST/
    """
    data=mne.io.read_raw_edf("F:/FYP/scalp dataset/PN00/{0}.edf".format(rec_num))
    time,raw_data = data.times,data.get_data()
# you can get the metadata included in the file and a list of all channels:
    info = data.info
    channels = data.ch_names
    c_num=20
    output=raw_data[c_num]
    channel_name=channels[c_num]
    #output=signal.resample_poly(output, down=360, up=250)
    return output,channel_name,time
    # ecg_signal=raw_record.p_signal
    # ecg_samples=np.arange(len(ecg_signal),dtype=int)
    # return raw_record,ecg_signal,ecg_samples

#------------records----------------------
#PN00-1,PN00-2,PN00-3,PN00-4,PN00-5
Y,ch_name,time=open_record("PN00-1")
Y2=mov_avg_filt(Y)
Y3=bandpass_filt(Y2)
Y4=denoise_wavelet(Y3, wavelet='db6', mode='soft', wavelet_levels=3, method='BayesShrink', rescale_sigma='True')


fig, axs = plt.subplots()
fig.suptitle('EEG templates')
show_wave(Y,time)
show_wave(Y2,time)
show_wave(Y3,time)
show_wave(Y4,time)


plt.show()