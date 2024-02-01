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
import seaborn as sns
import pandas as pd
import sys

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
    
    #y=np.multiply(np.multiply(y2,y3),y4)
    y=(y2&y3)|(y2&y4)|(y3&y4)
    #checking if y contains only 0s
    if np.any(y) and (str(i+1) in Files):
        valid_ans[str(i+1)]=np.repeat(y,fs)

valid_ans_keys=list(valid_ans.keys())
Channels=[[i] for i in range(19)] #zenodo

X=np.empty((len(Channels),))

record_num=int(sys.argv[1])
X,t=open_record_z(rec_num=valid_ans_keys[record_num],frs=fs,out="all-channels")

#normalizing X
mu_X=np.mean(X,axis=1)
sigma_X=np.std(X,axis=1)
X=(X.T-np.array([mu_X])).T/np.array([sigma_X]).T
clean_array=bf.appy_basics(X,btype="bandpass")
beta_val=0.3
k_1=0.1
k_2=5
#clean_arra_new=sp.eeg.ATAR(clean_array.T, wv='db3', winsize=128, beta=beta_val, thr_method='ipr', OptMode='soft', verbose=0,k1=k_1,k2=k_2)

#computing variation of mutual information with the change of k_2
k_2_list=np.array([round(i/10,2) for i in range(1,11)])
k_2_for_labels=["k= "+str(i) for i in k_2_list]
beta_val_list=np.array([0.1,0.2,0.3,0.4,0.5])
Mutual_information_art={}
Mutual_information_nart={}
Correlation_art={}
Correlation_nart={}
duration=120#in seconds

for beta in beta_val_list:
    MI_for_beta=[]
    MI_for_beta_nart=[]
    CC_for_beta=[]
    CC_for_beta_nart=[]
    print("beta=",beta)
    for k_2 in k_2_list:
        #clean_arra_new=sp.eeg.ATAR(clean_array[:,:duration*fs].T, wv='db3', winsize=128, beta=beta, thr_method='ipr', OptMode='soft', verbose=0,k1=k_1,k2=k_2).T
        t0=time.time()
        ref_clean_arr= sp.eeg.ICA_filtering(clean_array[:,:duration*fs].T,verbose=0,ICA_method="fastica",winsize=128).T
        clean_array_with_artifacts=sp.eeg.ATAR(clean_array[:,:duration*fs].T, wv='db3', winsize=128, beta=beta, thr_method='ipr', OptMode='soft', verbose=0,k1=k_1,k2=k_2).T
        clean_arra_no_artefacts=sp.eeg.ATAR(ref_clean_arr.T, wv='db3', winsize=128, beta=beta, thr_method='ipr', OptMode='soft', verbose=0,k1=k_1,k2=k_2).T
        MI_for_beta_k2=[]
        MI_for_beta_nart_k2=[]
        CC_for_beta_k2=[]
        CC_for_beta_nart_k2=[]
        for i in range(21):
            MI_for_beta_k2.append(calc_MI(clean_array[i,:duration*fs],clean_array_with_artifacts[i,:],bins=10))
            MI_for_beta_nart_k2.append(calc_MI(ref_clean_arr[i,:duration*fs],clean_arra_no_artefacts[i,:],bins=10))
            CC_for_beta_k2.append(np.corrcoef(clean_array[i,:duration*fs],clean_array_with_artifacts[i,:])[0,1])
            CC_for_beta_nart_k2.append(np.corrcoef(ref_clean_arr[i,:duration*fs],clean_arra_no_artefacts[i,:])[0,1])
        MI_for_beta.append(np.array(MI_for_beta_k2))
        MI_for_beta_nart.append(np.array(MI_for_beta_nart_k2))
        CC_for_beta.append(np.array(CC_for_beta_k2))
        CC_for_beta_nart.append(np.array(CC_for_beta_nart_k2))
    Mutual_information_art[beta]=np.array(MI_for_beta)
    Mutual_information_nart[beta]=np.array(MI_for_beta_nart)
    Correlation_art[beta]=np.array(CC_for_beta)
    Correlation_nart[beta]=np.array(CC_for_beta_nart)

#plotting the variation of mutual information and correlation. In each of the cases the mean and standard deviation of the values are calculated before plotting. std is plotted as an error bar

#reconstructing the dictionary into a 1-D array
def construnct_1D_dic(Data_dic):
    for beta in Data_dic:
        k_MI_CC=[]
        for k_2 in range(len(k_2_list)):
            #add mean and variance corresponding to k value
            k_MI_CC.append([k_2_list[k_2],np.mean(Data_dic[beta][k_2]),np.std(Data_dic[beta][k_2])])
        Data_dic[beta]=k_MI_CC

    return Data_dic

Mutual_information_art=construnct_1D_dic(Mutual_information_art)
Mutual_information_nart=construnct_1D_dic(Mutual_information_nart)
Correlation_art=construnct_1D_dic(Correlation_art)
Correlation_nart=construnct_1D_dic(Correlation_nart)

def make_df(Data_dic):
    df_list = []
    for beta, arr in Data_dic.items():
        # Create a dataframe for each beta value
        Col=["k_2","mean","std"]
        df = pd.DataFrame(arr, columns=Col)
        df["beta"] = beta
        df_list.append(df)
    # Concatenate the dataframes
    df = pd.concat(df_list, ignore_index=True)
    return df

df_MI_art=make_df(Mutual_information_art)
df_MI_nart=make_df(Mutual_information_nart)
df_CC_art=make_df(Correlation_art)
df_CC_nart=make_df(Correlation_nart)

#combining the dataframes
df_MI_art["type"]="artifacts"
df_MI_nart["type"]="no artifacts"
df_CC_art["type"]="artifacts"
df_CC_nart["type"]="no artifacts"
df_MI=pd.concat([df_MI_art,df_MI_nart],ignore_index=True)
df_CC=pd.concat([df_CC_art,df_CC_nart],ignore_index=True)

#plotting the dataframes grouped by artefacts and no artefacts
sns.set_theme()
#sns.set_style("ticks")

#plotting barchart for the mutual information, grouped by artefacts and no artefacts, for different values of beta mean should be the height and var should be the error bar
ax1=sns.catplot(data=df_MI, hue="type",x="k_2", y="mean",palette="colorblind")
plt.title("Variation of mutual information with k_2 - record {0} and for duration {1}s".format(valid_ans_keys[record_num],duration))
plt.savefig("MI_artefactual {0}.png".format(valid_ans_keys[record_num]),bbox_inches='tight',dpi=300,pad_inches=1)

ax2=sns.catplot(data=df_CC, hue="type",x="k_2", y="mean",palette="colorblind")
plt.title("Variation of correlation with k_2 - record {0} and for duration {1}s".format(valid_ans_keys[record_num],duration))
plt.savefig("CC_artefactual {0}.png".format(valid_ans_keys[record_num]),bbox_inches='tight',dpi=300,pad_inches=1)