import sys, os 
import numpy as np
from data_processing import readin, create_outname

inputfile = sys.argv[1]

X, Y, names, features, experiments = readin(inputfile, 'None', assign_region = False)
if ',' in inputfile:
    inputfiles = inputfile.split(',')
    inputfile = inputfiles[0]
    for inp in inputfiles[1:]:
        inputfile = create_outname(inp, inputfile, lword = 'and')

np.savez_compressed(inputfile+'.npz', seqfeatures = (X , np.array(list('ACGT'))), genenames = names)
