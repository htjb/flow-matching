from flowmatch.flowmatch import FlowMatch
from lsbi.stats import mixture_normal
import numpy as np
import matplotlib.pyplot as plt

dist = mixture_normal([0,0], [[0,0],[0,0]], 
                      np.array([[[1, 0.99],[0.99, 1]],
                                [[1, -0.99],[-0.99, 1]]]))
samples = dist.rvs(10000)

"""plt.scatter(samples[:,0], samples[:,1])
plt.show()"""

fm = FlowMatch(samples, 2, 3, 100)
fm.train(epochs=100)

samps = fm.sample(1000, time_samples=100)
plt.scatter(samps[-1, :, 0], samps[-1, :, 1])
plt.show()

lps = fm.log_prob(samples)

print(samps.shape)

for i in range(samps.shape[0]):
    plt.plot(samps[i, :, 0], samps[i, :, 1], '.', c='k', alpha=0.2)
    plt.xlim(samps[:, :, 0].min(), samps[:, :, 0].max())
    plt.ylim(samps[:, :, 1].min(), samps[:, :, 1].max())
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('example/flowmatch_{:03d}.png'.format(i))
    plt.close()