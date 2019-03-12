import numpy as np
import matplotlib.pyplot as plt
import ghalton
import sobol
import time

from scipy.stats import invgauss, norm


N = 5000

bin_width = 0.1

# tic = time.time()
# n1 = np.random.normal(0, 1, N)
# toc = time.time() - tic
# print(toc)
#
# tic = time.time()
# sequencer = ghalton.GeneralizedHalton(1)
# point = sequencer.get(N)
# n2 = norm.ppf(point)
# toc = time.time() - tic
# print(toc)

# n3 = sobol.i4_sobol(200, )

# plt.figure()
# plt.hist(n1, bins=np.arange(min(n1), max(n1) + bin_width, bin_width))
# plt.show()
# plt.figure()
# plt.hist(n2, bins=np.arange(min(n1), max(n1) + bin_width, bin_width))
# plt.show()


