import numpy as np
from scipy import signal

def appy_basics(input,btype,frs):
    if btype=='highpass':
        a,b=signal.butter(5, 1, btype='highpass', fs=frs, output='ba')
    elif btype=='bandpass':
        a,b=signal.cheby2(5, 40, [1, 40], btype='bandpass', fs=frs, output='ba')
    elif btype=='lowpass':
        a,b=signal.butter(5, 40, btype='lowpass', fs=frs, output='ba')
    filtered_output=signal.filtfilt(a,b,input)
    #notch filter for 50 Hz and 100Hz
    a,b=signal.iirnotch(50,30,frs)
    filtered_output=signal.filtfilt(a,b,filtered_output)
    return filtered_output