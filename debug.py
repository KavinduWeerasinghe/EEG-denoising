import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 6*np.pi, 600)
y = np.sin(x)

# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
line1,=ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma
line2,=ax.plot(x,np.cos(x),'b-')

for phase in np.linspace(0, 10*np.pi, 500):
    line1.set_ydata(np.sin(x + phase))
    line2.set_ydata(np.cos(x+phase))
    fig.canvas.draw()
    fig.canvas.flush_events()