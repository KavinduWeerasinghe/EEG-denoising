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

def bandpass_filt(Y,freq1=0.5,freq2=100):
    sos = signal.cheby1(2, 1, Wn=[freq1,freq2], btype='bandpass', fs=512, output='sos')
    filtered = signal.sosfilt(sos, Y)
    return filtered

def make_bands(Y):
    #delta
    delta = bandpass_filt(Y,1,4)
    theta = bandpass_filt(Y,4,8)
    alpha = bandpass_filt(Y,8,13)
    beta = bandpass_filt(Y,13,30)
    gamma = bandpass_filt(Y,30,100)
    return delta,theta,alpha,beta,gamma

def show_wave(r,samples,multiple=False):
    if multiple>=0:
        axs[multiple].plot(samples,r)
    else:
        axs.plot(samples,r)
        
def open_record(rec_fol: str, rec_num : str)->tuple:
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
    data=mne.io.read_raw_edf("F:/FYP/Sienna Scalp dataset/files/sienna-scalp-eeg/1.0.0/{0}/{1}.edf".format(rec_fol,rec_num))  # noqa: E501
    time,raw_data = data.times,data.get_data()
# you can get the metadata included in the file and a list of all channels:
    info = data.info
    channels = data.ch_names
    c_num=2
    output=raw_data[c_num]
    channel_name=channels[c_num]
    #output=signal.resample_poly(output, down=360, up=250)
    return output,channel_name,time
    # ecg_signal=raw_record.p_signal
    # ecg_samples=np.arange(len(ecg_signal),dtype=int)
    # return raw_record,ecg_signal,ecg_samples

#------------records----------------------
#PN00-1,PN00-2,PN00-3,PN00-4,PN00-5
Y,ch_name,time=open_record("PN00","PN00-3")
Y2=mov_avg_filt(Y)
#Y3=bandpass_filt(Y2)
Y3=denoise_wavelet(Y2, wavelet='db6', mode='soft', wavelet_levels=3, method='BayesShrink', rescale_sigma='True')
delta,theta,alpha,beta,gamma=make_bands(Y3)

fig, axs = plt.subplots(6)
fig.suptitle('EEG templates')
first_few,lenth=False,2000
if first_few:
    show_wave(Y[:lenth],time[:lenth],0)
    show_wave(Y2[:lenth],time[:lenth],0)
    show_wave(Y3[:lenth],time[:lenth],0)
    show_wave(delta[:lenth],time[:lenth],1)
    show_wave(theta[:lenth],time[:lenth],2)
    show_wave(alpha[:lenth],time[:lenth],3)
    show_wave(beta[:lenth],time[:lenth],4)
    show_wave(gamma[:lenth],time[:lenth],5)
else:
    show_wave(Y,time,0)
    show_wave(Y2,time,0)
    show_wave(Y3,time,0)

    show_wave(delta,time,1)
    show_wave(theta,time,2)
    show_wave(alpha,time,3)
    show_wave(beta,time,4)
    show_wave(gamma,time,5)

plt.show()