import matplotlib.pyplot as plt
import numpy as np
from scipy import signal,fft
from Read.Read import open_record
from Filtering import Basic_filters as bf
from sklearn.decomposition import FastICA,PCA
import pandas as pd
import glob
import mne
from mne_icalabel import label_components

dfs = []
#"D:/FYP/CHB-MIT/{0}/{0}_{1}.edf"
for i_csv in sorted(glob.glob("D:/FYP/archive/SMNI_CMI_TEST/Data5.csv")):
    print(i_csv)
    dfs.append(pd.read_csv(i_csv))
    
df = pd.concat(dfs)
df.loc[df['sensor position']=='FP1', 'sensor position'] = 'Fp1'
df.loc[df['sensor position']=='FP2', 'sensor position'] = 'Fp2'
df.loc[df['sensor position']=='CZ', 'sensor position'] = 'Cz'
df.loc[df['sensor position']=='PZ', 'sensor position'] = 'Pz'
df.loc[df['sensor position']=='FZ', 'sensor position'] = 'Fz'

eeg_data = {}
for sensor in df['sensor position'].unique():
    eeg_data[sensor] = df[df['sensor position']==sensor]['sensor value'].values
    
chans_1020 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 
              'C4', 'P3', 'Pz', 'P4', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2']

eeg = []
channel_names = []
for chan in eeg_data.keys():
    if chan in chans_1020:
        channel_names.append(chan)
        eeg.append(eeg_data[chan])
        
data = np.stack(eeg)
sf = 256

# make 1020
mne_info = mne.create_info(
                    channel_names,
                    sf,
                    ch_types=['eeg']*len(channel_names)
)

mne_raw = mne.io.RawArray(data, mne_info)

standard_1020 = mne.channels.make_standard_montage('standard_1020')
mne_raw.set_montage(standard_1020)

# Set filter parameters
lowCut = 4 # Hz
highCut = 120 # Hz
freqNotch = 60 # Hz

# Apply bandpass and notch filter
mne_raw.filter(lowCut, highCut, fir_design='firwin')
mne_raw.notch_filter(freqNotch, fir_design='firwin')

#plot the channels
mne_raw.plot(scalings='auto',clipping=None)

ica_obj = mne.preprocessing.ICA(
                    n_components=0.9,
                    method='infomax',
                    max_iter="auto",
                    random_state=1,
                    fit_params=dict(extended=True)).fit(mne_raw)

ic_labels = label_components(mne_raw, ica_obj, method="iclabel")
labels = ic_labels["labels"]
exclude = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]

#printing the labels
for i in range(len(labels)):
    print("{0}: {1}".format(i,labels[i]))

ica = ica_obj.get_sources(mne_raw).get_data()
print(ica.shape)

#remove the components 4 and 7
ica_obj.exclude = exclude
mne_raw = ica_obj.apply(mne_raw)

#plot the channels
mne_raw.plot(scalings='auto',clipping=None)

ica_obj.plot_sources(mne_raw)

ica_obj.plot_components(picks=None,show=True, inst=mne_raw)


#ica_obj.plot_properties(mne_raw, picks=[0], show=True)

# https://labeling.ucsd.edu/tutorial/overview more on IC labeling

