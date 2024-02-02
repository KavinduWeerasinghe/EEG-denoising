from pymatreader import read_mat
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import kurtosis,skew,entropy
import eeglib
import joblib
import time
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QPushButton
import pyqtgraph as pg
import scipy
from Read.Read_zenodo import open_record as open_record_z
from Filtering import Basic_filters as bf
import threading
import sys
import time
from sklearn.decomposition import PCA, FastICA

def hj_mobility(data,axis_=1):
    return np.sqrt(np.var(np.diff(data,axis=axis_),axis=axis_)/np.var(data,axis=axis_))

def hj_complexity(data,axis_=1):
    return hj_mobility(np.diff(data,axis=axis_),axis_=axis_)/hj_mobility(data,axis_=axis_)

def predict_artefact_class(model,data_i):
    #normalise the data_i array
    #t1=time.time()
    mu_i=np.mean(data_i,axis=1)
    mu_i=mu_i.reshape(mu_i.shape[0],1)
    sigma_i=np.std(data_i,axis=1)
    sigma_i=sigma_i.reshape(sigma_i.shape[0],1)
    data_i=(data_i-mu_i)/sigma_i

    #mobility
    hj_mob_i=hj_mobility(data_i)
    hj_mob_i=hj_mob_i.reshape(hj_mob_i.shape[0],1)

    #complexity
    hj_comp_i=hj_complexity(data_i)
    hj_comp_i=hj_comp_i.reshape(hj_comp_i.shape[0],1)

    #skew
    skewness_i=skew(data_i,axis=1)**2
    skewness_i=skewness_i.reshape(skewness_i.shape[0],1)

    #kurtosis
    kurtosis_i=kurtosis(data_i,axis=1)**2
    kurtosis_i=kurtosis_i.reshape(kurtosis_i.shape[0],1)
    print(hj_mob_i.shape,skewness_i.shape,kurtosis_i.shape)
    data_i=np.hstack((hj_mob_i,skewness_i,kurtosis_i,hj_comp_i)).T

    #predict
    pred=model.predict(data_i.T)
    #print(model.predict_proba(data_i.T))
    #t2=time.time()
    #print("time taken: ",t2-t1)

    return pred

annotations=scipy.io.loadmat("D:/FYP/Zenodo EEG/zenodo_eeg/annotations_2017.mat")
fs=256

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
    y=y2&y3&y4
    #checking if y contains only 0s
    if np.any(y) and (str(i+1) in Files):
        valid_ans[str(i+1)]=np.repeat(y,fs)

def signal_array(raw_data):
    ch1=raw_data[0]-raw_data[5]
    ch2=raw_data[5]-raw_data[7]
    ch3=raw_data[0]-raw_data[2]
    ch4=raw_data[2]-raw_data[7]
    ch5=raw_data[1]-raw_data[3]
    ch6=raw_data[3]-raw_data[8]
    ch7=raw_data[1]-raw_data[6]
    ch8=raw_data[6]-raw_data[8]
    ch9=raw_data[5]-raw_data[2]
    ch10=raw_data[2]-raw_data[4]
    ch11=raw_data[4]-raw_data[3]
    ch12=raw_data[3]-raw_data[6]
    return np.array([ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12])

valid_ans_keys=list(valid_ans.keys())
Channels=[[i] for i in range(19)] #zenodo

X=np.empty((len(Channels),))

record_num=20
X,t=open_record_z(rec_num=valid_ans_keys[record_num],frs=fs,out="all-channels")
X=signal_array(X)

#normalizing X
mu_X=np.mean(X,axis=1)
sigma_X=np.std(X,axis=1)
X=(X.T-np.array([mu_X])).T/np.array([sigma_X]).T
clean_array=bf.appy_basics(X,btype="bandpass",frs=fs)
EEG=clean_array
labels=valid_ans[valid_ans_keys[record_num]]

seconds=2
updated_array,labels_array=np.zeros((12,256*seconds)),np.zeros((12,256*seconds))
app = QApplication([])
win = pg.GraphicsLayoutWidget(show=True)
# win.setBackground('w')
plot_c1=[]
plot_c2=[]

channel_names=["Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2","Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"]
IC_val=["IC1","IC2","IC3","IC4","IC5","IC6","IC7","IC8","IC9","IC10","IC11","IC12"]

for i in range(6):
    plot_c1.append(win.addPlot(title=channel_names[i]))
    plot_c2.append(win.addPlot(title=channel_names[i+6]))
    #plot_c1[i].setRange(xRange=[0,384],yRange=[-1e-4,1e-4])
    #plot_c2[i].setRange(xRange=[0,384],yRange=[-1e-4,1e-4])
    win.nextRow()

graph_c1 = []
graph_c2 = []
seizure_c1=[]
seizure_c2=[]
ICA_c1=[]
ICA_c2=[]

for i in range(6):
    graph_c1.append(plot_c1[i].plot(pen='y',width=10))
    graph_c2.append(plot_c2[i].plot(pen='y',width=10))
    seizure_c1.append(plot_c1[i].plot(pen='r',width=10))
    seizure_c2.append(plot_c2[i].plot(pen='r',width=10))
    ICA_c1.append(plot_c1[i].plot(pen='g',width=10))
    ICA_c2.append(plot_c2[i].plot(pen='g',width=10))

def update_plot():
    for i in range(6):
        graph_c1[i].setData(y=EEG1[i])
        graph_c2[i].setData(y=EEG1[i+6])
        seizure_c1[i].setData(y=label_EEG1[i])
        seizure_c2[i].setData(y=label_EEG1[i+6])
        ICA_c1[i].setData(y=ica_EEG1[i])
        ICA_c2[i].setData(y=ica_EEG1[i+6])

def break_ICA(array):
    ica = FastICA(n_components=12, whiten="unit-variance")
    S_ = ica.fit_transform(array.T)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix
    return S_,A_


interval=0.06
l,u=0,int(interval*fs)
c=0
labels_array=np.zeros((12,256*seconds))
ica_array=np.zeros((12,256*seconds))
model=joblib.load("D:/FYP/Code/Practise Code/random_forest.joblib")

def loop():
    global l,u,c,EEG1,label_EEG1,ica_EEG1
    #fig,ax=plt.subplots(6,2,figsize=(20,20))
    while True:
        updated_array[:,:fs*seconds-int(interval*fs)]=updated_array[:,int(interval*fs):]
        updated_array[:,fs*seconds-int(interval*fs):]=EEG[:,l:u]
        labels_array[:,:fs*seconds-int(interval*fs)]=labels_array[:,int(interval*fs):]
        labels_array[:,fs*seconds-int(interval*fs):]=labels[l:u]*EEG[:,l:u]
        #ica_array[:,:fs*seconds-int(interval*fs)]=ica_array[:,int(interval*fs):]
        ica_array[:,:]=break_ICA(updated_array)[0].T
        EEG1=updated_array
        label_EEG1=labels_array
        ica_EEG1=ica_array
        if c>=2:
            predictions=predict_artefact_class(model,ica_array)
            print(predictions)
        update_plot()
        c+=1
        l+=int(interval*fs)
        u+=int(interval*fs)
        if u>EEG.shape[1]:
            print("done")
            break
        time.sleep(0.5)
    #plt.pause(0.001)

timer = QtCore.QTimer()
timer.timeout.connect(update_plot)



if __name__ == '__main__':

    t1 = threading.Thread(target=loop)
    t1.start()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QApplication.instance().exec_()