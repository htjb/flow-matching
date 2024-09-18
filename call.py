from flowmatch.flowmatch import FlowMatch
from lsbi.stats import mixture_normal
import numpy as np
import matplotlib.pyplot as plt

dist = mixture_normal([0,0], [[0,0],[0,0]], 
                      np.array([[[1, 0.99],[0.99, 1]],[[1, -0.99],[-0.99, 1]]]))
samples = dist.rvs(10000)

"""plt.scatter(samples[:,0], samples[:,1])
plt.show()"""

fm = FlowMatch(samples, 2, 3, 100)
fm.train()

samps = fm.sample(1000)
print(samps)