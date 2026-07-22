import numpy as np
import matplotlib.pyplot as plt


theta = np.arange(0, 360.)*np.pi/180.

a0 = 140220.
e0 = 0.00235

da_arr = [0.,100., -100.]
de_arr = [0., 0.0007, -0.0007]

colors = ['red', 'green', 'blue']

for de in de_arr:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    i = 0
    for da in da_arr:
        r = (a0+ da)*(1-de**2)/(1 + (e0 +de)*np.cos(theta))
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        plt.plot(x,y, color = colors[i])
        i += 1
    plt.show()
    
        