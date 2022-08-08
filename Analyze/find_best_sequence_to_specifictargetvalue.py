import numpy as np
import sys, os
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from compare_expression_distribution import read_separated

peaks = np.genfromtxt(sys.argv[1], dtype = str,delimiter = ',')
outname = os.path.splitext(sys.argv[1])[0]
tracknames = np.array(open(sys.argv[1], 'r').readline().strip().strip('#').split(',')[1:])
peaknames = peaks[:,0]
peaks = peaks[:,1:].astype(float)

target_tracks, tset_tracks = read_separated(sys.argv[2])

target_values = sys.argv[3]
if ',' in target_values:
    target_values = target_values.split(',')
    if len(target_values) != len(target_tracks) and tset_tracks is not None:
        tsetvalues = []
        for t, tval in enumerate(target_values):
            tsetvalues.append(np.ones(len(tset_tracks[t]))*float(tval))
        target_values = np.concatenate(tsetvalues)
else:
    target_values = [target_values]
target_values = np.array(target_values, dtype = float)

if '--target_set' in sys.argv:
    tset = np.genfromtxt(sys.argv[sys.argv.index('--target_set')+1], dtype = str)
    outname = os.path.splitext(sys.argv[sys.argv.index('--target_set')+1])[0]
    tmask = np.isin(peaknames, tset)
    peaknames = peaknames[tmask]
    peaks = peaks[tmask]

outname += sys.argv[2] + '_' +sys.argv[3]

print(target_values, target_tracks)    
distance = cdist(target_values.reshape(1,-1), peaks[:,target_tracks], 'euclidean').flatten()/len(target_tracks)

distance_sort = np.argsort(distance)

cutoff = 0.225
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.argsort(distance_sort), distance)
ax.plot(ax.get_xlim(), [cutoff, cutoff])
ax.set_xticks([len(distance[distance<cutoff])])

plt.show()



distance_sort = distance_sort[distance[distance_sort]<cutoff]
for e, d in enumerate(distance_sort):
    print(e,peaknames[d], round(distance[d],2), np.around(peaks[d][target_tracks],1))
print(outname, int(np.sum(distance[distance_sort]<cutoff)))
np.savetxt(outname +'list_genes.txt',peaknames[distance_sort], fmt = '%s')


