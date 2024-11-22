import numpy as np
import sys, os
from scipy.stats import pearsonr

heatmap = np.genfromtxt(sys.argv[1], dtype = str)
filt = sys.argv[2]

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

if '--groups' in sys.argv:
    groups = np.genfromtxt(sys.argv[sys.argv.index('--groups')+1], dtype = str, delimiter = '\t')
    group = []
    for c, cellt in enumerate(celltype):
        group.append(groups[list(groups[:,0]).index(cellt),1])
elif '--headerpart' in sys.argv:
    part = int(sys.argv[sys.argv.index('--headerpart')+1])
    group = [cellt.split('.')[part] for cellt in celltype]


mask = np.array(group) == filt
print('left', np.sum(mask), 'of', len(group))


np.savetxt(outstart+'.'+filt+'.dat', np.append(heatnames, heatmap[:,mask], axis = 1), header = ' '.join(np.append(header, xticklabels[mask])), fmt = '%s')


maskedx = xticklabels[mask]
for m in maskedx:
    print(m)
