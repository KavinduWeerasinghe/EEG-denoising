import matplotlib.pyplot as plt
import numpy as np
from Read.Read import open_record
from asrpy import asr_calibrate, asr_process,clean_windows
from Filtering import Basic_filters as bf

fs=250
fs=250
Folders=["chb01","chb02","chb03"]
Files=["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","35","36","37","38","39","40","41","42","43","44","45","46"]

mode="all_channels"

if mode=="BNC_config":
    Channels=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[20,21,22]] #mode = BNC_config
else:
    Channels=[[i] for i in range(23)]                                              #mode = all_channels

output,channel_name,time,annotation=open_record(Folders[0],Files[14],Channels[4],mode)

#plot the output
fig1, axs1 = plt.subplots()
axs1.plot(time,output)
axs1.set_title("Original Signal")
axs1.set_xlabel("Time (s)")
axs1.set_ylabel("Amplitude")
axs1.grid(True)

time=90  #seconds
start=3 #minutes
starting_sample=int(start*60*fs)
ending_samples=starting_sample+int(time*fs)
samples=ending_samples-starting_sample

#getting all of the outputs
X=np.empty((len(Channels),samples))

for i in range(len(Channels)):
    output,channel_name,time,annotation=open_record(Folders[0],Files[14],Channels[i],mode)
    filtered_output=bf.appy_basics(output)
    X[i,:]=filtered_output[starting_sample:ending_samples]
    #X[i,:]=np.array(output)[starting_sample:ending_samples]

time=time[starting_sample:ending_samples]

X=X

print(X.shape)

#plot the 23 channels
fig2, axs2 = plt.subplots(5)
for i in range(5):
    j=i*4
    axs2[i].plot(time,X[j,:])
    axs2[i].set_title("Channel {}".format(j+1))
    axs2[i].set_xlabel("Time (s)")
    axs2[i].set_ylabel("Amplitude")
    axs2[i].grid(True)


#convert X to a numpy array
X=np.array(X)

pre_cleaned, _ = clean_windows(X, fs, max_bad_chans=0.3)
M, T = asr_calibrate(pre_cleaned, fs, cutoff=5)
clean_array = asr_process(X, fs, M, T)

#delay componsate for 0.25 seconds
delay=-0.25
delay_samples=int(delay*fs)
clean_array=np.roll(clean_array,delay_samples,axis=1)

print(clean_array.shape)

#plot the output on ax2
for i in range(5):
    j=i*4
    axs2[i].plot(time,clean_array[j,:])
    axs2[i].set_title("Channel {}".format(j+1))
    axs2[i].set_xlabel("Time (s)")
    axs2[i].set_ylabel("Amplitude")
    axs2[i].grid(True)


plt.show()

