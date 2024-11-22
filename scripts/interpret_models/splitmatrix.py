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
for u, ud in enumerate(udata):
    print(ud)
    outname = outstart+'.'+ud+os.path.splitext(sys.argv[1])[1]
    mask = np.where(datatype == ud)[0]
    np.savetxt(outname, np.append(heatnames, heatmap[:,mask], axis = 1), header = ' '.join(np.append(header, celltype[mask])), fmt = '%s')
    print(outname)


