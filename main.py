import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import numpy as np





# generate some random data
np.random.seed(12345)
nstars = 10000
xs = np.random.uniform(0, 1, size=nstars)
ys = np.random.uniform(0, 1, size=nstars)




fig, ax = plt.subplots()
ax.scatter(xs, ys)




def get_all_tris(ids):
    perms = list(itertools.combinations(ids, 3))
    return perms
    
allperms = set()
N=4
for ii in tqdm(range(len(xs))):
    dists = (xs-xs[ii])**2 + (ys-ys[ii])**2
    closest = np.argsort(dists).tolist()[:N+1]

    perms = [tuple(sorted(p)) for p in get_all_tris(closest)]
    for p in perms:
        allperms.add(p)
#        if p not in allperms:
#            allperms.append(p)

for p in allperms:
    ax.plot([xs[p[0]], xs[p[1]]], [ys[p[0]], ys[p[1]]])
    ax.plot([xs[p[0]], xs[p[2]]], [ys[p[0]], ys[p[2]]])
    ax.plot([xs[p[1]], xs[p[2]]], [ys[p[1]], ys[p[2]]])
#
plt.show()



