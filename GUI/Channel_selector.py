import  tkinter as tk
import matplotlib.pyplot as plt
from tkinter import ttk

window=tk.Tk()
window.title('Channel Selector')
window.geometry('1500x900')

def plot_one_plot(time,EEG,ICA,channel,annotation):
    fig_tk,axs_tk=plt.subplots()
    axs_tk.plot(time,EEG[:,0])
    axs_tk.plot(time,ICA[:,channel])
    axs_tk.set_title("ICA Component {}".format(channel))
    axs_tk.set_xlabel("Time (s)")
    axs_tk.set_ylabel("Amplitude")
    axs_tk.grid(True)
    for j in range(len(annotation)):
        axs_tk.axvline(x=annotation[j],color="red")
    plt.show()

def display_tk_window(time,EEG,ICA,annotation):
    #add a drop-down box to select the channel
    tk.Label(window,text='Select your Channel').place(x=630,y=850)
    combo=ttk.Combobox(
        state='readonly',
        values=[
            'Channel 1',
            'Channel 2',
            'Channel 3',
            'Channel 4',
            'Channel 5',
            'Channel 6',
            'Channel 7',
            'Channel 8',
            'Channel 9'
        ]
    )
    combo.place(x=750,y=850)
    values=['Channel 1',
            'Channel 2',
            'Channel 3',
            'Channel 4',
            'Channel 5',
            'Channel 6',
            'Channel 7',
            'Channel 8',
            'Channel 9']
    #add a button to plot the graph
    tk.Button(window,text='Plot',command=lambda: plot_one_plot(time,EEG,ICA,values.index(combo.get()),annotation)).place(x=1030,y=845)

    window.mainloop()

