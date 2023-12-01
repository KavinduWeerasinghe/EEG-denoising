from scipy import signal,stats
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import openpyxl

#function to implement the butterworth filter
def butterworth_filter(data, cutoff, f, order=5):
    """
    :param data: data to be filtered
    :param cutoff: cutoff frequency
    :param fs: sampling rate
    :param order: order of the filter, default is 5
    :return: filtered data
    """
    rp=0.5
    b, a = signal.butter(order, cutoff, btype='bandpass', analog=False,fs=f)
    y = signal.filtfilt(b, a, data)
    return y

#function to implement the chebyshev type 2 filter
def chebyshev2_filter(data, cutoff, f, order=5):
    """
    :param data: data to be filtered
    :param cutoff: cutoff frequency
    :param fs: sampling rate
    :param order: order of the filter, default is 5
    :return: filtered data
    """
    rp=0.5
    # Get the filter coefficients
    b, a = signal.cheby2(order, rp, cutoff, btype='bandpass', analog=False,fs=f)
    y = signal.filtfilt(b, a, data)
    return y

#function to chebeyshev type 1 filter
def chebyshev1_filter(data, cutoff, f, order=5):
    """
    :param data: data to be filtered
    :param cutoff: cutoff frequency
    :param fs: sampling rate
    :param order: order of the filter, default is 5
    :return: filtered data
    """
    rp=0.5
    # Get the filter coefficients
    b, a = signal.cheby1(order, rp, cutoff, btype='bandpass', analog=False,fs=f)
    y = signal.filtfilt(b, a, data)
    return y

#function to implement the elliptic filter
def elliptic_filter(data, cutoff, f, order=5):
    """
    :param data: data to be filtered
    :param cutoff: cutoff frequency
    :param fs: sampling rate
    :param order: order of the filter, default is 5
    :return: filtered data
    """
    
    rp,rs = 0.5,5
    # Get the filter coefficients
    b, a = signal.ellip(order, rp,rs, Wn=cutoff, btype='bandpass', analog=False,fs=f)
    y = signal.filtfilt(b, a, data)
    return y

#function to implement the bessel filter
def bessel_filter(data, cutoff, f, order=5):
    """
    :param data: data to be filtered
    :param cutoff: cutoff frequency
    :param fs: sampling rate
    :param order: order of the filter, default is 5
    :return: filtered data
    """

    # Get the filter coefficients
    b, a = signal.bessel(order, cutoff, btype='bandpass', analog=False,fs=f)
    y = signal.filtfilt(b, a, data)
    return y

#function to normalized MSE between 2 1-D EEG arrays
def nmse(x, y):
    A= np.linalg.norm(x - y) / np.linalg.norm(x)
    #return A in db
    return 10*np.log10(A)

def signaltonoise_mod(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def signaltonoise(original,filtered, axis=0, ddof=0):
    data = filtered
    singleChannel = data
    norm = singleChannel / (max(np.amax(singleChannel), -1 * np.amin(singleChannel)))
    return signaltonoise_mod(norm)
    
    # mse = np.mean((original - filtered) ** 2)
    # if(mse == 0):  # MSE is zero means no noise is present in the signal .
    #               # Therefore PSNR have no importance.
    #     return 100
    # max_pixel = 255.0
    # psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    #return psnr

#function to run the filters on the the input EEG variable for butterworth, chebyshev type 2, chebyshev type 1, elliptic, bessel
#return should be nMSE for each filter
#cutoff frequencies are 0.5Hz and 40Hz, sampling rate is fs
def run_filters(EEG,fs):
    cutoff=[2,30]
    butterworth=butterworth_filter(EEG,cutoff,fs)
    chebyshev2=chebyshev2_filter(EEG,cutoff,fs)
    chebyshev1=chebyshev1_filter(EEG,cutoff,fs)
    elliptic=elliptic_filter(EEG,cutoff,fs)
    bessel=bessel_filter(EEG,cutoff,fs)
    butterworthnMSE=nmse(EEG,butterworth)
    chebyshev2nMSE=nmse(EEG,chebyshev2)
    chebyshev1nMSE=nmse(EEG,chebyshev1)
    ellipticnMSE=nmse(EEG,elliptic)
    besselnMSE=nmse(EEG,bessel)
    #calculate the MI for each filter
    butterworthMI=mutual_info_regression(EEG.reshape(-1,1),butterworth.reshape(-1,1))
    chebyshev2MI=mutual_info_regression(EEG.reshape(-1,1),chebyshev2.reshape(-1,1))
    chebyshev1MI=mutual_info_regression(EEG.reshape(-1,1),chebyshev1.reshape(-1,1))
    ellipticMI=mutual_info_regression(EEG.reshape(-1,1),elliptic.reshape(-1,1))
    besselMI=mutual_info_regression(EEG.reshape(-1,1),bessel.reshape(-1,1))
    return butterworthnMSE,chebyshev2nMSE,chebyshev1nMSE,ellipticnMSE,besselnMSE,butterworthMI, chebyshev2MI, chebyshev1MI, ellipticMI, besselMI

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

#channel to combine several channels eg: FP1-T3 from FP1-F7 and F7-T3
def combine_channels(channel1,channel2,channel_name1,channel_name2):
    combined_channel=channel1+channel2
    #modifed channel with F7 removed
    mod_channel_name=channel_name1+","+channel_name2
    return combined_channel,mod_channel_name

#function to write values to an excel file
def write_result(folder_name,file_name,channel_name,parameter,filtered_values)->None:
    path="F:/FYP/Code/Stats.xlsx"
    book = openpyxl.load_workbook(path)
    book.active=book[parameter]
    Sheet=book.active
    
    #column headers are already there. find the empty row
    """for i in range(2,2000):
        C_cell=Sheet["A{0}".format(i)]
        record=C_cell.value
        if record is None:
            Cell=i
            break"""
    #covert filtered values to a list and flatten it.
    filtered_values=list(filtered_values)
    print(filtered_values)
    Results=[folder_name,file_name,channel_name]+filtered_values
    Sheet.append(Results)
    print("Results written to excel file"+"\n"*10)
    book.save(path)


#code to calculate nMSE for all the channels in the EEG data (29 channels)
folder_name="PN01"
file_names=["PN01-1","PN01-2","PN01-3","PN01-4","PN01-5"]

fig, axs = plt.subplots(1,1)
fig.suptitle('Normalized mean square error for all the channels')

#open another figure to plot the MI for each filter
# fig1, axs1 = plt.subplots(1,1)
# fig1.suptitle('Mutual information for all the channels')

Channels=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[20,21,22]]
for file_name in file_names:
    for k,i in enumerate(Channels):
        EEG,channel_name,time=open_record(folder_name,file_name,i[0])
        for j in i[1:]:
            EEG_,channel_name_,time_=open_record(folder_name,file_name,j)
            EEG,channel_name=combine_channels(EEG,EEG_,channel_name,channel_name_)
        butterworthnMSE,chebyshev2nMSE,chebyshev1nMSE,ellipticnMSE,besselnMSE,butterworthMI, chebyshev2MI, chebyshev1MI, ellipticMI, besselMI=run_filters(EEG,250)  # noqa: E501
        #plot the values on a single plot
        Filters=["butterworth","chebyshev2","chebyshev1","elliptic","bessel"]
        nMSE=[butterworthnMSE,chebyshev2nMSE,chebyshev1nMSE,ellipticnMSE,besselnMSE]
        MI=[butterworthMI[0],chebyshev2MI[0],chebyshev1MI[0],ellipticMI[0],besselMI[0]]
        axs.scatter(Filters,nMSE,label=channel_name)
        axs.set_xlabel("Filters")
        axs.set_ylabel("nMSE")
        
        #write the values to an excel file
        #write_result(folder_name,file_name,channel_name,"nMSE",nMSE)
        
        #plot the MI values on a corresponding plot
        #axs1.scatter(Filters,MI,label=channel_name)
        #axs1.set_xlabel("Filters")
        #axs1.set_ylabel("MI")
        
        #write the values to an excel file
        #write_result(folder_name,file_name,channel_name,"MI",MI)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()


#choosen channels
"""
1. FP1-T3           2. FP2-T4
3. FP1-C3           4. FP2-C4
5. C3-T3            6. C4-T4
7. C3-Cz            8. C4-Cz
9. T3-O1            10. T4-O2
11. C3-O1           12. C4-O2
"""