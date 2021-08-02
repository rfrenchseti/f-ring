import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import factorial

def poisson(k,mean):
    return mean**k * np.exp(-mean) / factorial(k)

mean = np.arange(0., 3., 0.001)

########################
# Voyager
########################

# Probability of observing >= 2 clumps twice

# Probabiliy of (not) observing exactly 0 clumps or exactly 1 clumps
prob_ge2 = 1. - poisson(0,mean) - poisson(1,mean)

# Probability of observing two or more independent clumps
# WHY IS THIS TWO OR MORE?
prob_vgr = prob_ge2**2

########################
# Cassini
########################

prob_eq0 = poisson(0,mean)
prob_eq1 = poisson(1,mean)
prob_eq2 = poisson(2,mean)

for n in range(5,11):

    # Probability of never observing a bright clump
    # 0 clumps in each of N slots
    prob_never_of_n = prob_eq0**n

    # Probability of observing exactly one bright clump
    # 1 clump in the first slot, then 0 clumps...or...
    # 1 clump in the second slot, otherwise 0...or...
    # N times
    prob_once_of_n = n * prob_eq0**(n-1) * prob_eq1

    # Probability of observing exactly two bright clumps
    # 1 clump in first slot, then 1 clump in second slot, then 0s...or...
    # 1 clump in first slot, then 1 clump in third slot, then 0s...or...
    # First clump can be in one of N locations
    # Second clump can be in one of N-1 locations but divide by 2 because
    #   the two clumps are interchangeable
    # Zeros in remaining locations
    # We can also have two clumps at the same time in any of N slots
    prob_twice_of_n = n*(n-1)/2. * prob_eq0**(n-2) * prob_eq1**2 + \
                      n * prob_eq0**(n-1) * prob_eq2

    prob_cas = prob_never_of_n + prob_once_of_n + prob_twice_of_n

    plt.plot(mean, prob_vgr * prob_cas * 100, label=str(n))

plt.xlabel('Expected number of large ECs $\lambda$')
plt.ylabel('Probability (%)')
plt.legend()

plt.show()

