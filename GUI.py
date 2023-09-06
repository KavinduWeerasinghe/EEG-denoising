import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# generate root
root = ctk.CTk()
root.geometry("800x400")

def make_GUI(fig):
    #add a button to close the window
    quit_button = ctk.CTkButton(root, text="Quit", command=quit)
    quit_button.place(relx=0.9, rely=0.9)
    
    # generate the figure and plot object which will be linked to the root element
    canvas = FigureCanvasTkAgg(fig,master=root)
    canvas.draw()
    canvas.get_tk_widget().place(relx=0.15, rely=0.15)
    
    # initiate the window
    root.mainloop()
    
    #close
    root.destroy()
    
def quit():
    global root
    root.quit()