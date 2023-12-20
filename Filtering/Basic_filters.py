import numpy as np
from scipy import signal

def appy_basics(input):
    a,b=signal.cheby2(4, 40, [0.5, 100], btype='bandpass', fs=250, output='ba')
    filtered_output=signal.filtfilt(a,b,input)
    #notch filter for 50 Hz and 100Hz
    a,b=signal.iirnotch(50,30,250)
    filtered_output=signal.filtfilt(a,b,filtered_output)
    return filtered_output