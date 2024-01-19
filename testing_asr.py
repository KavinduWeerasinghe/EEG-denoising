import matplotlib.pyplot as plt
import numpy as np
from Read.Read import open_record
from asrpy import asr_calibrate, asr_process,clean_windows
from Filtering import Basic_filters as bf
from Read.Read_zenodo import open_record as open_record_z
import time as Time
import spkit as sp
from scipy.signal import periodogram as psd

def applyasr(X_5min,fs=250):
    pre_cleaned, _ = clean_windows(X_5min, fs, max_bad_chans=0.7)
    M, T = asr_calibrate(pre_cleaned, fs, cutoff=5)
    clean_array = asr_process(X_5min, fs, M, T)

    delay=-0.25
    delay_samples=int(delay*fs)
    clean_array=np.roll(clean_array,delay_samples,axis=1)

    return clean_array

fs=128
#chb-mit
#Folders=["chb01","chb02","chb03"]
#Files=["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","35","36","37","38","39","40","41","42","43","44","45","46"]
#zenodo
Files=[str(i) for i in range(2,25)]+[str(i) for i in range(26,62)]+[str(i) for i in range(63,80)] 
mode="all_channels"  #all_channels, BNC_config, or custom

if mode=="BNC_config":
    Channels=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[20,21,22]] #mode = BNC_config
else:
    #mode = all_channels
    Channels=[[i] for i in range(23)] #chb-mit
    Channels=[[i] for i in range(19)] #zenodo                       

#chb-mit
#output,channel_name,time,annotation=open_record(Folders[0],Files[14],Channels[4],mode,plot_graph=True)

#zenodo
output,channel_name,time=open_record_z(Files[14],Channels[4],mode,frs=fs)

#plot the output
fig1, axs1 = plt.subplots()
axs1.plot(time,output)
axs1.set_title("Original Signal")
axs1.set_xlabel("Time (s)")
axs1.set_ylabel("Amplitude")
axs1.grid(True)

samples=len(output)
#getting all of the outputs
X=np.empty((len(Channels),samples))

for i in range(len(Channels)):
    #output,channel_name,time,annotation=open_record(Folders[0],Files[14],Channels[i],mode) #chb-mit
    output,channel_name,time=open_record_z(Files[14],Channels[i],mode,frs=fs) #zenodo
    #filtered_output=bf.appy_basics(output)
    X[i,:]=output
    #X[i,:]=np.array(output)[starting_sample:ending_samples]

print(X.shape)

X=np.array(X)

#X=X*(3000000)

#nomralizing X
X_mus=np.mean(X,axis=1)
X_sigmas=np.std(X,axis=1)
X=(X-np.array([X_mus]).T)/np.array([X_sigmas]).T

#plot the signal after normalization
fig3, axs3 = plt.subplots()
axs3.plot(time,X[0,:])
axs3.set_title("Normalized Signal")
axs3.set_xlabel("Time (s)")
axs3.set_ylabel("Amplitude")
axs3.grid(True)
plt.show()

start0=300
seconds=5
minutes=1
start0=start0*fs
X_5min=X[:,start0:start0+minutes*seconds*fs]
time_5min=time[start0:start0+minutes*seconds*fs]
clean_array=bf.appy_basics(X_5min,btype='highpass')
clean_array_new = clean_array.T
beta_val=0.3
k_1=0.01
k_2=1
clean_array_new= sp.eeg.ATAR(clean_array_new.copy(), wv='db3', winsize=128, beta=beta_val, thr_method='ipr', OptMode='soft', verbose=0,k1=k_1,k2=k_2)
clean_array_new=clean_array_new.T
fx_psd,X_psd=psd(X_5min,fs=250)
f_clean_psd,clean_array_psd=psd(clean_array,fs=250)
f_new_psd,clean_array_new_psd=psd(clean_array_new,fs=250)
#clean_array=applyasr(X_5min,fs)

plt.ion()

#plotting the output
fig2, axs2 = plt.subplots(5,2)
#plotting the 19 channels
print(X_5min.shape,clean_array.shape)

channel_list=[1,3,5,7,9]

lin1,=axs2[0,0].plot(time_5min,X_5min[channel_list[0],:],color="blue")
lin2,=axs2[0,0].plot(time_5min,clean_array[channel_list[0],:],color="red")
lin3,=axs2[1,0].plot(time_5min,X_5min[channel_list[1],:],color="blue")
lin4,=axs2[1,0].plot(time_5min,clean_array[channel_list[1],:],color="red")
lin5,=axs2[2,0].plot(time_5min,X_5min[channel_list[2],:],color="blue")
lin6,=axs2[2,0].plot(time_5min,clean_array[channel_list[2],:],color="red")
lin7,=axs2[3,0].plot(time_5min,X_5min[channel_list[3],:],color="blue")
lin8,=axs2[3,0].plot(time_5min,clean_array[channel_list[3],:],color="red")
lin9,=axs2[4,0].plot(time_5min,X_5min[channel_list[4],:],color="blue")
lin10,=axs2[4,0].plot(time_5min,clean_array[channel_list[4],:],color="red")
lin21,=axs2[0,0].plot(time_5min,clean_array_new[channel_list[0],:],color="green")
lin22,=axs2[1,0].plot(time_5min,clean_array_new[channel_list[1],:],color="green")
lin23,=axs2[2,0].plot(time_5min,clean_array_new[channel_list[2],:],color="green")
lin24,=axs2[3,0].plot(time_5min,clean_array_new[channel_list[3],:],color="green")
lin25,=axs2[4,0].plot(time_5min,clean_array_new[channel_list[4],:],color="green")

#plotting the psd of the signals between 1 and 80 Hz
lin11,=axs2[0,1].semilogy(fx_psd,np.sqrt(X_psd[channel_list[0],:]),color="blue")
lin12,=axs2[1,1].semilogy(fx_psd,np.sqrt(X_psd[channel_list[1],:]),color="blue")
lin13,=axs2[2,1].semilogy(fx_psd,np.sqrt(X_psd[channel_list[2],:]),color="blue")
lin14,=axs2[3,1].semilogy(fx_psd,np.sqrt(X_psd[channel_list[3],:]),color="blue")
lin15,=axs2[4,1].semilogy(fx_psd,np.sqrt(X_psd[channel_list[4],:]),color="blue")
lin16,=axs2[0,1].semilogy(f_clean_psd,np.sqrt(clean_array_psd[channel_list[0],:]),color="red")
lin17,=axs2[1,1].semilogy(f_clean_psd,np.sqrt(clean_array_psd[channel_list[1],:]),color="red")
lin18,=axs2[2,1].semilogy(f_clean_psd,np.sqrt(clean_array_psd[channel_list[2],:]),color="red")
lin19,=axs2[3,1].semilogy(f_clean_psd,np.sqrt(clean_array_psd[channel_list[3],:]),color="red")
lin20,=axs2[4,1].semilogy(f_clean_psd,np.sqrt(clean_array_psd[channel_list[4],:]),color="red")
lin30,=axs2[0,1].semilogy(f_new_psd,np.sqrt(clean_array_new_psd[channel_list[0],:]),color="green")
lin31,=axs2[1,1].semilogy(f_new_psd,np.sqrt(clean_array_new_psd[channel_list[1],:]),color="green")
lin32,=axs2[2,1].semilogy(f_new_psd,np.sqrt(clean_array_new_psd[channel_list[2],:]),color="green")
lin33,=axs2[3,1].semilogy(f_new_psd,np.sqrt(clean_array_new_psd[channel_list[3],:]),color="green")
lin34,=axs2[4,1].semilogy(f_new_psd,np.sqrt(clean_array_new_psd[channel_list[4],:]),color="green")
for i in range(5):
    axs2[i,1].set_xlim([1,80])
    axs2[i,1].set_ylim([10**(-4),1])

for start in range(start0+1,start0+1400000,fs):
    X_5min=X[:,start:start+minutes*seconds*fs]
    time_5min=time[start:start+minutes*seconds*fs]
    clean_array=bf.appy_basics(X_5min,btype='highpass')
    clean_array_new = clean_array.T
    clean_array_new = sp.eeg.ATAR(clean_array_new.copy(),wv='db4', winsize=128, beta=beta_val,thr_method='ipr',OptMode='soft', verbose=0,k1=k_1,k2=k_2)
    clean_array_new=clean_array_new.T
    fx_psd,X_psd=psd(X_5min,fs=250)
    f_clean_psd,clean_array_psd=psd(clean_array,fs=250)
    f_new_psd,clean_array_new_psd=psd(clean_array_new,fs=250)
    #clean_array=applyasr(X_5min,fs)

    #delay componsate for 0.25 seconds

    #updating the data values
    plus,minus=-2.5,2.5
    #plus,minus=1,-1
    lin1.set_ydata(X_5min[channel_list[0],:])
    lin2.set_ydata(clean_array[channel_list[0],:])
    axs2[0,0].set_ylim([plus,minus])
    lin3.set_ydata(X_5min[channel_list[1],:])
    lin4.set_ydata(clean_array[channel_list[1],:])
    axs2[1,0].set_ylim([plus,minus])
    lin5.set_ydata(X_5min[channel_list[2],:])
    lin6.set_ydata(clean_array[channel_list[2],:])
    axs2[2,0].set_ylim([plus,minus])
    lin7.set_ydata(X_5min[channel_list[3],:])
    lin8.set_ydata(clean_array[channel_list[3],:])
    axs2[3,0].set_ylim([plus,minus])
    lin9.set_ydata(X_5min[channel_list[4],:])
    lin10.set_ydata(clean_array[channel_list[4],:])
    axs2[4,0].set_ylim([plus,minus])
    lin21.set_ydata(clean_array_new[channel_list[0],:])
    lin22.set_ydata(clean_array_new[channel_list[1],:])
    lin23.set_ydata(clean_array_new[channel_list[2],:])
    lin24.set_ydata(clean_array_new[channel_list[3],:])
    lin25.set_ydata(clean_array_new[channel_list[4],:])
    lin1.set_xdata(time_5min)
    lin2.set_xdata(time_5min)
    lin3.set_xdata(time_5min)
    lin4.set_xdata(time_5min)
    lin5.set_xdata(time_5min)
    lin6.set_xdata(time_5min)
    lin7.set_xdata(time_5min)
    lin8.set_xdata(time_5min)
    lin9.set_xdata(time_5min)
    lin10.set_xdata(time_5min)
    lin21.set_xdata(time_5min)
    lin22.set_xdata(time_5min)
    lin23.set_xdata(time_5min)
    lin24.set_xdata(time_5min)
    lin25.set_xdata(time_5min)
    for i in range(5):
        axs2[i,0].set_xlim([time_5min[0],time_5min[-1]])

    #plotting the psd of the signals between 1 and 80 Hz
    lin11.set_ydata(np.sqrt(X_psd[channel_list[0],:]))
    lin12.set_ydata(np.sqrt(X_psd[channel_list[1],:]))
    lin13.set_ydata(np.sqrt(X_psd[channel_list[2],:]))
    lin14.set_ydata(np.sqrt(X_psd[channel_list[3],:]))
    lin15.set_ydata(np.sqrt(X_psd[channel_list[4],:]))
    lin16.set_ydata(np.sqrt(clean_array_psd[channel_list[0],:]))
    lin17.set_ydata(np.sqrt(clean_array_psd[channel_list[1],:]))
    lin18.set_ydata(np.sqrt(clean_array_psd[channel_list[2],:]))
    lin19.set_ydata(np.sqrt(clean_array_psd[channel_list[3],:]))
    lin20.set_ydata(np.sqrt(clean_array_psd[channel_list[4],:]))
    lin30.set_ydata(np.sqrt(clean_array_new_psd[channel_list[0],:]))
    lin31.set_ydata(np.sqrt(clean_array_new_psd[channel_list[1],:]))
    lin32.set_ydata(np.sqrt(clean_array_new_psd[channel_list[2],:]))
    lin33.set_ydata(np.sqrt(clean_array_new_psd[channel_list[3],:]))
    lin34.set_ydata(np.sqrt(clean_array_new_psd[channel_list[4],:]))
    for i in range(5):
        axs2[i,1].set_xlim([1,80])
        axs2[i,1].set_ylim([10**(-4),10])

    fig2.canvas.draw()
    fig2.canvas.flush_events()

    #updating the plot
    
    #print("Current j: {0}".format(start))


