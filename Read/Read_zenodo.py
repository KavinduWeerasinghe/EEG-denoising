import mne
from scipy import signal
import numpy
#"D:\FYP\Zenodo EEG\zenodo_eeg\eeg2.edf"
def open_record(rec_num:str,Channel_list=[[i] for i in range(19)],mode='BNC_config')->tuple:
    file="D:/FYP/Zenodo EEG/zenodo_eeg/eeg{0}.edf".format(rec_num)
    data=mne.io.read_raw_edf(file)
    time,raw_data = data.times,data.get_data()
    info = data.info
    fs=info["sfreq"]
    channels = data.ch_names
    Output=[0 for i in range(len(Channel_list))]
    channel_name=[0 for i in range(len(Channel_list))]
    for i,c_num in enumerate(Channel_list):
        Output[i]=raw_data[c_num]
        channel_name[i]=channels[c_num]
    if mode=="BNC_config":
        Output,channel_name=combine_channels(Output[0],Output[1],channel_name[0],channel_name[1])
    else:
        Output=Output[0]
        channel_name=channel_name[0]
    Output=signal.resample_poly(Output, down=fs, up=250)
    time=numpy.linspace(0, len(Output)/250, num=len(Output))
    return Output,channel_name,time

def combine_channels(channel1,channel2,channel_name1,channel_name2):
    """Combine two channels into one."""
    combined_channel=channel1+channel2
    mod_channel_name=channel_name1+","+channel_name2
    return combined_channel,mod_channel_name