import mne
from scipy import signal
import numpy

#function to read a .txt file
def read_txt(file_name: str,file_num)->list:
    path="D:/FYP/CHB-MIT/{0}/{0}-summary.txt".format(file_name)
    with open(path) as f:
        lines = f.readlines()
    for k,i in enumerate(lines):
        file_num_="{0}_{1}.edf".format(file_name,file_num)
        if file_num_ in i:
            Num_sz=lines[k+3].split(":")[1].strip()
            Num_sz=int(Num_sz)
            if Num_sz==1:
                start=int((lines[k+4].split(":")[1].strip()).split(" ")[0])
                end=int((lines[k+5].split(":")[1].strip()).split(" ")[0])
                start=int(start*250/360)
                end=int(end*250/360)
                return start,end
            else:
                return 0,0
    return lines

def open_record(rec_fol: str, rec_num : str,Channel_list:list,mode="BNC_config")->tuple:
    data=mne.io.read_raw_edf("D:/FYP/CHB-MIT/{0}/{0}_{1}.edf".format(rec_fol,rec_num))  # noqa: E501
    time,raw_data = data.times,data.get_data()
    info = data.info
    channels = data.ch_names
    output=[0 for i in range(len(Channel_list))]
    channel_name=[0 for i in range(len(Channel_list))]
    for i,c_num in enumerate(Channel_list):
        print(c_num)
        output[i]=raw_data[c_num]
        channel_name[i]=channels[c_num]
    if mode=="BNC_config":
        output,channel_name=combine_channels(output[0],output[1],channel_name[0],channel_name[1])
    else:
        output=output[0]
        channel_name=channel_name[0]
    output=signal.resample_poly(output, down=360, up=250)
    time=numpy.linspace(0, len(output)/250, num=len(output))
    start,end = read_txt(rec_fol,rec_num)
    return output,channel_name,time,[start,end]

def combine_channels(channel1,channel2,channel_name1,channel_name2):
    """Combine two channels into one."""
    combined_channel=channel1+channel2
    mod_channel_name=channel_name1+","+channel_name2
    return combined_channel,mod_channel_name