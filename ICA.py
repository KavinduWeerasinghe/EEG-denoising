import numpy as np
import matplotlib.pyplot as plt
import spkit as sp
from spkit.data import load_data

X,ch_names = load_data.eegSample()
fs = 128
Xf = sp.filter_X(X,band=[0.5], btype='highpass',fs=fs,verbose=0)
print(Xf.shape)
t = np.arange(Xf.shape[0])/fs
plt.figure(figsize=(12,4))
plt.plot(t,Xf+np.arange(-7,7)*200)
plt.xlim([t[0],t[-1]])
plt.xlabel('time (sec)')
plt.yticks(np.arange(-7,7)*200,ch_names)
plt.grid()
plt.title('Xf: 14 channel - EEG Signal (filtered)')
plt.show()

XR = sp.eeg.ICA_filtering(Xf.copy(),verbose=0,winsize=128)

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(t,Xf+np.arange(-7,7)*200)
plt.xlim([t[0],t[-1]])
#plt.xlabel('time (sec)')
plt.yticks(np.arange(-7,7)*200,ch_names)
plt.grid()
plt.title('X: Filtered signal',fontsize=16)
plt.subplot(222)
plt.plot(t,XR+np.arange(-7,7)*200)
plt.xlim([t[0],t[-1]])
#plt.xlabel('time (sec)')
plt.yticks(np.arange(-7,7)*200,ch_names)
plt.grid()
plt.title('XR: Corrected Signal',fontsize=16)
#plt.show()
#plt.figure(figsize=(12,5))
plt.subplot(223)
plt.plot(t,(Xf-XR)+np.arange(-7,7)*200)
plt.xlim([t[0],t[-1]])
plt.xlabel('time (s)')
plt.yticks(np.arange(-7,7)*200,ch_names)
plt.grid()
plt.title('Xf - XR: Difference \n(removed signal)',fontsize=16)
# #plt.show()
plt.subplot(224)
plt.plot(t,Xf[:,0],label='Xf')
plt.plot(t,XR[:,0],label='XR')
#plt.plot(t,Xf[:,0]-XR[:,0])
plt.xlim([t[0],t[-1]])
plt.xlabel('time (s)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()