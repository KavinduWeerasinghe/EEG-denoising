from tkinter import *
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib import pyplot as plt
from filters import butterworth_filter,chebyshev2_filter

#function to make the GUI for an input figure
def make_second_GUI(fig,axis,time,Y):
    window = Tk()
    window.title("EEG Analysis")
    window.geometry('500x500')
    plot_button = Button(master = window,   # noqa: F841, F405
                        command = make_GUI,
                        height = 2, 
                        width = 10,
                        text = "Plot")
    canvas = FigureCanvasTkAgg(fig,
                                 master = window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    toolbar.update()
    canvas.get_tk_widget().pack()
    
    Button(window, text="Quit", command=window.quit,bg="red",fg="white").pack()  # noqa: F405, E501
    
    #add a dropdown to select the filter
    variable = StringVar(window)  # noqa: F405
    variable.set("butterworth") # default value
    
    #check the selected filter from dropdown
    def ok(variable):
        if variable=="butterworth":
            Y2=butterworth_filter(Y,30,250)
        elif variable=="chebyshev2":
            Y2=chebyshev2_filter(Y,30,250)
        else:
            print("Wrong filter")
        #update the canvas on the new Y2
        # fig2, axs2 = plt.subplots(1,1)
        # fig2.suptitle('Filtered EEG')
        axis.plot(time,Y2,label=variable)
        axis.set_xlabel("time")
        plt.legend()
        fig.canvas.draw_idle()
        
    #run ok() depending on the selected filter
    w = OptionMenu(window, variable, "butterworth", "chebyshev2",command=ok)  # noqa: F405
    w.pack()
    
    window.mainloop()

def make_GUI(fig,time,Y):
    canvas = FigureCanvasTkAgg(fig,
                               master = window)  
    canvas.draw()
  
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
  
    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    toolbar.update()
  
    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()
    
    #function to pop up the filtered EEG
    def pop_up():
        fig2, axs2 = plt.subplots(1,1)
        fig2.suptitle('Filtered EEG')
        
        default="butterworth"
        Y3=butterworth_filter(Y,30,250)
        
        axs2.plot(time,Y3,label="butterworth")
        axs2.set_xlabel("time")
        axs2.set_ylabel("amplitude")
        axs2.set_title("Filtered EEG")
        
        #add legend
        plt.legend()
        
        #show the plot in GUI
        make_second_GUI(fig2,axs2,time,Y)
         
    #button to pop up the filtered EEG
    Button(window, text="Filtered EEG", command=pop_up,bg="green",fg="white").pack()
    
    #button to close the window in red color and crossing sign
    Button(window, text="Quit", command=window.quit,bg="red",fg="white").pack()
    
    window.mainloop()

# Create a window
window = Tk()

# Create a title
window.title("EEG Analysis")

# Create dimensions
window.geometry('500x500')

plot_button = Button(master = window, 
                     command = make_GUI,
                     height = 2, 
                     width = 10,
                     text = "Plot")


