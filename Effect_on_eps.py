import matplotlib.pyplot as plt
import numpy as np
from Read.Read import open_record
from asrpy import asr_calibrate, asr_process,clean_windows
from Filtering import Basic_filters as bf
from Read.Read_zenodo import open_record as open_record_z
import time
import spkit as sp
from scipy.signal import periodogram as psd
import scipy.io
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def applyasr(X_5min,fs=250):
    pre_cleaned, _ = clean_windows(X_5min, fs, max_bad_chans=0.7)
    M, T = asr_calibrate(pre_cleaned, fs, cutoff=15)
    clean_array = asr_process(X_5min, fs, M, T)

    delay=-0.25
    delay_samples=int(delay*fs)
    clean_array=np.roll(clean_array,delay_samples,axis=1)

    return clean_array

annotations=scipy.io.loadmat("D:/FYP/Zenodo EEG/zenodo_eeg/annotations_2017.mat")
fs=128

valid_ans={}
Files=[str(i) for i in range(2,25)]+[str(i) for i in range(26,62)]+[str(i) for i in range(63,80)] 
#plotting the first 5 annotations
for i in range(79):
    y0=annotations["annotat_new"][0]
    y1=y0[i]
    y2=y1[0]
    y3=y1[1]
    y4=y1[2]
    
    y=np.multiply(np.multiply(y2,y3),y4)

    #checking if y contains only 0s
    if np.any(y) and (str(i+1) in Files):
        valid_ans[str(i+1)]=np.repeat(y,fs)

valid_ans_keys=list(valid_ans.keys())
Channels=[[i] for i in range(19)] #zenodo

X=np.empty((len(Channels),))

record_num=0
X,t=open_record_z(rec_num=valid_ans_keys[record_num],frs=fs,out="all-channels")

#normalizing X
mu_X=np.mean(X,axis=1)
sigma_X=np.std(X,axis=1)
X=(X.T-np.array([mu_X])).T/np.array([sigma_X]).T
clean_array=bf.appy_basics(X,btype="highpass")
beta_val=0.3
k_1=0.1
k_2=5
#clean_arra_new=sp.eeg.ATAR(clean_array.T, wv='db3', winsize=128, beta=beta_val, thr_method='ipr', OptMode='soft', verbose=0,k1=k_1,k2=k_2)

#computing variation of mutual information with the change of k_2
k_2_list=[0.5,0.8,1,2,3,4,5]
beta_val_list=[0.1,0.2,0.3,0.4,0.5]
Mutual_information={}
Correlation={}
duration= 20 #in seconds
for k_2 in k_2_list:
    clean_arra_new=sp.eeg.ATAR(clean_array[:,:duration*fs].T, wv='db3', winsize=128, beta=beta_val, thr_method='ipr', OptMode='soft', verbose=0,k1=k_1,k2=k_2).T
    Mutual_information[k_2]=[]
    Correlation[k_2]=[]
    for i in range(21):
        Mutual_information[k_2].append(calc_MI(clean_array[i,:duration*fs],clean_arra_new[i,:],bins=10))
        #Mutual_information[k_2].append(mutual_info_score(clean_array[i,:duration*fs],clean_arra_new[i,:]))
        Correlation[k_2].append(np.corrcoef(clean_array[i,:duration*fs],clean_arra_new[i,:])[0,1])

#plotting the variation of mutual information and correlation. In each of the cases the mean and standard deviation of the values are calculated before plotting
fig, axs = plt.subplots(2,1,figsize=(10,10))
axs[0].plot(k_2_list,np.mean(np.array(list(Mutual_information.values())),axis=1))
axs[0].set_title("Variation of Mutual Information with k_2")
axs[0].set_xlabel("k_2")
axs[0].set_ylabel("Mutual Information")
axs[0].grid(True)

axs[1].plot(k_2_list,np.mean(np.array(list(Correlation.values())),axis=1))
axs[1].set_title("Variation of Correlation with k_2")
axs[1].set_xlabel("k_2")
axs[1].set_ylabel("Correlation")
axs[1].grid(True)
plt.show()

