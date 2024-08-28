import numpy as np
import sys, os

intfile=sys.argv[1]
exfile=sys.argv[2]
outname=sys.argv[3]

ex = np.genfromtxt(exfile, dtype = str, skip_header = 1)
kt = np.genfromtxt(intfile, dtype = str, skip_header = 1)
excell = np.array(open(exfile, 'r').readline().strip('#').strip().split())
ktcell = np.array(open(intfile, 'r').readline().strip('#').strip().split())

exg = ex[:,0]
ktg = kt[:,0]
ex = ex[:,1:].astype(float)
kt = kt[:,1:].astype(float)
if np.array_equal(excell, ktcell):
    if not np.array_equal(exg, ktg):
        maskex = np.argsort(exg)[np.isin(np.sort(exg), ktg)]
        maskkt = np.argsort(ktg)[np.isin(np.sort(ktg), exg)]
        exg, ktg, ex, kt = exg[maskex], ktg[maskkt], ex[maskex], kt[maskkt]
    mask = ex == 0
    print(np.shape(mask), np.where(mask))
    kt[mask] = 0
    # log intronic minus log exonic counts
    if '--log2' in sys.argv:
        kt = np.log2(kt)
        ex = np.log2(ex)
    kd = kt - ex
    np.savetxt(outname, np.append(exg.reshape(-1,1), kd.astype(str), axis =1), header = '\t'.join(excell), fmt = '%s', delimiter = '\t')

else:
    print('cellnotthesame')

