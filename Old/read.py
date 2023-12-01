import numpy as np
from scipy import signal
import mne
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from GUI import make_GUI

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
        
def open_annotations(file,number):
    with open(file) as f:
        lines = f.readlines()
        if str(number) in lines:
            print(lines)
    return None

def open_record(rec_fol: str, rec_num : str,c_num=2)->tuple:
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
    data=mne.io.read_raw_edf("D:/FYP/CHB-MIT/chb01/chb01_01.edf".format(rec_fol,rec_num))  # noqa: E501
    time,raw_data = data.times,data.get_data()
    #open_annotations("F:/FYP/Sienna Scalp dataset/files/sienna-scalp-eeg/1.0.0/{0}/Seizures-list-{1}.txt".format(rec_fol,rec_fol),rec_num)  # noqa: E501
    # you can get the metadata included in the file and a list of all channels:
    info = data.info
    channels = data.ch_names
    output=raw_data[c_num]
    channel_name=channels[c_num]
    #output=signal.resample_poly(output, down=360, up=250)
    return output,channel_name,time
    # ecg_signal=raw_record.p_signal
    # ecg_samples=np.arange(len(ecg_signal),dtype=int)
    # return raw_record,ecg_signal,ecg_samples

#------------records----------------------
#PN00-1,PN00-2,PN00-3,PN00-4,PN00-5
Y,ch_name,time=open_record("PN01","PN01-1")
Y2=mov_avg_filt(Y)
#Y3=bandpass_filt(Y2)
Y3=denoise_wavelet(Y2, wavelet='db6', mode='soft', wavelet_levels=3, method='BayesShrink', rescale_sigma='True')
delta,theta,alpha,beta,gamma=make_bands(Y3)

fig, axs = plt.subplots(2,2)
fig.suptitle('EEG templates')
    
#plot the raw EEG
axs[0,0].plot(time,Y)
axs[0,0].set_xlabel("time")
axs[0,0].set_ylabel("amplitude")

#plot the alpha band
axs[0,1].plot(time,alpha)
axs[0,1].set_xlabel("time")
axs[0,1].set_ylabel("amplitude")

#plot the beta band
axs[1,0].plot(time,beta)
axs[1,0].set_xlabel("time")
axs[1,0].set_ylabel("amplitude")

#plot the gamma band
axs[1,1].plot(time,gamma)
axs[1,1].set_xlabel("time")
axs[1,1].set_ylabel("amplitude")

#give the name of the channel

plt.legend([ch_name,"moving average","denoised"])

#give all the title of the subplots
axs[0,0].set_title("Raw EEG")
axs[0,1].set_title("Alpha band")
axs[1,0].set_title("Beta band")
axs[1,1].set_title("Gamma band")

#show the plot in GUI
make_GUI(fig,time,Y)
