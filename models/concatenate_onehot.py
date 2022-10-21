import sys, os 
import numpy as np
from data_processing import readin, create_outname

inputfile = sys.argv[1]

mirror = False
if '--realign_input' in sys.argv:
    mirror = True

X, Y, names, features, experiments = readin(inputfile, 'None', assign_region = False, mirrorx = mirror)
if ',' in inputfile:
    inputfiles = inputfile.split(',')
    inputfile = inputfiles[0]
    for inp in inputfiles[1:]:
        inputfile = create_outname(inp, inputfile, lword = 'and')

if '--realign_input' in sys.argv:
    inputfile += 'algnbth'

np.savez_compressed(inputfile+'.npz', seqfeatures = (X , np.array(list('ACGT'))), genenames = names)
