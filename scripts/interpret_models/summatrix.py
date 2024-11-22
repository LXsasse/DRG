import numpy as np
import sys, os
from scipy.stats import pearsonr

heatmap = np.genfromtxt(sys.argv[1], dtype = str)
heatnames = heatmap[:,:2]
heatmap = heatmap[:, 2:]
header = open(sys.argv[1], 'r').readline().strip('#').strip()
xticklabels = np.array(header.split())[2:]
header = np.array(header.split())[:2]

datatype = []
celltype = []
for x, xt in enumerate(xticklabels):
    xts = xt.rsplit('.',1)
    datatype.append(xts[-1])
    celltype.append(xts[0])
datatype = np.array(datatype)
celltype = np.array(celltype)

outstart = os.path.splitext(sys.argv[1])[0]
if '--outname' in sys.argv:
    outstart = sys.argv[sys.argv.index('--outname')+1]

udata = np.unique(datatype)
sumheatmap = np.zeros((len(heatnames), len(udata)))
cvheatmap = np.zeros((len(heatnames), len(udata)))
maxheatmap = np.zeros((len(heatnames), len(udata)))
scaledheatmap = np.zeros((len(heatnames), len(udata)))
scaledmaxheatmap = np.zeros((len(heatnames), len(udata)))

for u, ud in enumerate(udata):
    print(ud)
    mask = np.where(datatype == ud)[0]
    sumheatmap[:,u] = np.mean(heatmap[:,mask].astype(float), axis = 1)
    cvheatmap[:,u] = np.std(heatmap[:,mask].astype(float), axis = 1)/np.mean(heatmap[:,mask].astype(float), axis = 1)
    
    amax = np.argmax(np.absolute(heatmap[:,mask].astype(float)), axis = 1)
    maxheatmap[:,u] = heatmap[:,mask][np.arange(len(heatmap)),amax].astype(float)
    
    scaledheat = heatmap[:,mask].astype(float)/np.amax(np.absolute(heatmap[:,mask].astype(float)))
    scaledheatmap[:,u] = np.mean(scaledheat, axis = 1)
    amax = np.argmax(np.absolute(scaledheat), axis = 1)
    scaledmaxheatmap[:,u] = scaledheat[np.arange(len(heatmap)),amax]

    
np.savetxt(outstart+'.meanmod.dat', np.append(heatnames, sumheatmap, axis = 1), header = ' '.join(np.append(header, udata)), fmt = '%s')
np.savetxt(outstart+'.cvmod.dat', np.append(heatnames, cvheatmap, axis = 1), header = ' '.join(np.append(header, udata)), fmt = '%s')
np.savetxt(outstart+'.scaledmeanmod.dat', np.append(heatnames, scaledheatmap, axis = 1), header = ' '.join(np.append(header, udata)), fmt = '%s')
np.savetxt(outstart+'.maxmod.dat', np.append(heatnames, maxheatmap, axis = 1), header = ' '.join(np.append(header, udata)), fmt = '%s')
np.savetxt(outstart+'.scaledmaxmod.dat', np.append(heatnames, scaledmaxheatmap, axis = 1), header = ' '.join(np.append(header, udata)), fmt = '%s')


