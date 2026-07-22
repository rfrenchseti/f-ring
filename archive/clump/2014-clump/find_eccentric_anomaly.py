import numpy as np
import math
import os
import scipy.optimize as sciopt
import pickle
import ringutil




eccen = 0.000235
num_longitudes = 18000

def keplers_equation_resid(p, M, e):
# p = (E)
# params = (M, e)
    return np.sqrt((M - (p[0]-e*np.sin(p[0])))**2)

# Mean anomaly M = n(t-tau)
# Find E, the eccentric anomaly, the angle from the center of the ellipse and the pericenter to a
# circumscribed circle at the place where the orbit is projected vertically.
def find_E(M, e):
    result = sciopt.fmin(keplers_equation_resid, (0.,), args=(M, e), disp=False, xtol=1e-20, ftol=1e-20)
    return result[0]


# These are really mean anomaly!!
longitude_list = 2.*np.pi/num_longitudes * np.array(range(num_longitudes))
        
        #find the Eccentric Anomaly
E_list = []
for m in longitude_list:
    E = find_E(m, eccen)
    E_list.append(E)
    
E_list = np.array(E_list)

file = os.path.join(ringutil.ROOT, 'eccentric_anom_arr_18000')
np.save(file, E_list)

