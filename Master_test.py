import matplotlib.pyplot as plt
import numpy as np
from scipy import signal,fft
from Read.Read import open_record
from Filtering import Basic_filters as bf
from sklearn.decomposition import FastICA
from GUI import Channel_selector as cs

fs=250
Folders=["chb01","chb02","chb03"]
Files=["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","35","36","37","38","39","40","41","42","43","44","45","46"]

Channels=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[20,21,22]]

output,channel_name,time,annotation=open_record(Folders[0],Files[14],Channels[4])

#plot the output
fig1, axs1 = plt.subplots()
axs1.plot(time,output)
axs1.set_title("Original Signal")
axs1.set_xlabel("Time (s)")
axs1.set_ylabel("Amplitude")
axs1.grid(True)

#getting all of the outputs
X=np.empty((len(Channels),len(output)))

for i in range(len(Channels)):
    output,channel_name,time,annotation=open_record(Folders[0],Files[14],Channels[i])
    filtered_output=bf.appy_basics(output)
    X[i,:]=filtered_output

X=X.T

print(X.shape)

sources=9
view_sources=5

#normalizing the output
output=(output-np.mean(output))/np.std(output)

ica=FastICA(n_components=sources,
            whiten=False)
ica_output=ica.fit_transform(X)

print(ica_output.shape)

cs.display_tk_window(time,X,ica_output,annotation)

#create subplots
fig, axs = plt.subplots(sources)

#figure 2 to include FFT
fig2, axs2 = plt.subplots(sources)

range_l,range_h=0,64000
#normalizing the ica output
ica_output=(ica_output-np.mean(ica_output))/np.std(ica_output)

for i in range(sources):
    #normalizing the ica output
    axs[i].plot(time[range_l:range_h],output[range_l:range_h])
    axs[i].plot(time[range_l:range_h],ica_output[range_l:range_h,i])
    axs[i].set_title("ICA Component {}".format(i+1))
    axs[i].set_xlabel("Time (s)")
    axs[i].set_ylabel("Amplitude")
    axs[i].grid(True)
    #for j in range(len(annotation)):
        #axs[i].axvline(x=annotation[j],color="red")
    #FFT
    f, t, Sxx = signal.spectrogram(ica_output[range_l:range_h,i], fs)
    axs2[i].semilogy(f, Sxx)
    axs2[i].set_title("FFT of ICA Component {}".format(i+1))
    axs2[i].set_xlabel("Frequency (Hz)")
    axs2[i].set_ylabel("PSD")
    axs2[i].grid(True)

#computing correlation of components with each channel
corr=np.empty((sources,len(Channels)))
for i in range(sources):
    for j in range(len(Channels)):
        corr[i,j]=np.corrcoef(ica_output[:,i],X[:,j])[0,1]
        print("Correlation of component {} with channel {} is {}".format(i+1,j+1,corr[i,j]))

#plotting the correlation in a single plot
fig3, axs3 = plt.subplots()
for i in range(sources):
    axs3.plot(corr[i,:],label="Component {}".format(i+1))
axs3.set_title("Correlation of components with channels")
axs3.set_xlabel("Channel")
axs3.set_ylabel("Correlation")
axs3.legend(["Component {}".format(i+1) for i in range(sources)])
axs3.grid(True)

plt.show()

