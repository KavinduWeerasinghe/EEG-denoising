import matplotlib.pyplot as plt
import numpy as np
from scipy import signal,fft
from Read.Read import open_record
from Filtering import Basic_filters as bf
from sklearn.decomposition import FastICA,PCA
from GUI import Channel_selector as cs

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

#getting all of the outputs
X=np.empty((len(Channels),len(output)))

for i in range(len(Channels)):
    output,channel_name,time,annotation=open_record(Folders[0],Files[14],Channels[i],mode)
    #filtered_output=bf.appy_basics(output)
    #X[i,:]=filtered_output
    X[i,:]=output

X=X.T

print(X.shape)

sources=9
view_sources=5

#normalizing the output
#output=(output-np.mean(output))/np.std(output)

pca=PCA(n_components=sources)
pca_output=pca.fit_transform(X)

print(pca.explained_variance_ratio_)

#plot the pca output
fig2, axs2 = plt.subplots(3,3)
for i in range(3):
    for j in range(3):
        axs2[i,j].plot(time,pca_output[:,i*3+j])
        axs2[i,j].set_title("PCA Component {}".format(i*3+j))
        axs2[i,j].set_xlabel("Time (s)")
        axs2[i,j].set_ylabel("Amplitude")
        axs2[i,j].grid(True)
        axs2[i,j].axhline(y=0,color="red")
        axs2[i,j].axvline(x=annotation[0],color="red")
        axs2[i,j].axvline(x=annotation[1],color="red")
        axs2[i,j].set_ylim(-0.001,0.001)

ica=FastICA(n_components=sources,
            whiten=False)
ica_output=ica.fit_transform(pca_output)


#plot the ica output but for a small timeframe eg: 10 seconds
fig3, axs3 = plt.subplots(3,3)
for i in range(3):
    for j in range(3):
        axs3[i,j].plot(time,ica_output[:,i*3+j])
        axs3[i,j].set_title("ICA Component {}".format(i*3+j))
        axs3[i,j].set_xlabel("Time (s)")
        axs3[i,j].set_ylabel("Amplitude")
        axs3[i,j].grid(True)
        axs3[i,j].axhline(y=0,color="red")
        axs3[i,j].axvline(x=annotation[0],color="red")
        axs3[i,j].axvline(x=annotation[1],color="red")
        axs3[i,j].set_ylim(-0.001,0.001)
        axs3[i,j].set_xlim(10,20)

plt.show()

cs.display_tk_window(time,X,ica_output,annotation)

plt.show()

