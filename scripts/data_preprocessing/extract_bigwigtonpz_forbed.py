'''
uses pyBigWig to read bigwig files and regturn activity in given regions for a
bed file
'''

import numpy as np
import sys, os
import pyBigWig
import sys, os
from drg_tools.sequence_utils import avgpool
from drg_tools.io_utils import isint

# TODO include this to io_utils and replace with pandas dataframe
def readbed(bedfile):
    bf = open(bedfile, 'r').readlines()
    bedfile = {'names':[], 'contigs':[], 'positions': []}
    for l, line in enumerate(bf):
        line = line.strip().split()
        bedfile['names'].append(line[0])
        bedfile['contigs'].append(line[1])
        bedfile['positions'].append([int(line[2]),int(line[3])])
    bedfile['names'] = np.array(bedfile['names'])
    bedfile['contigs'] = np.array(bedfile['contigs'])
    bedfile['positions'] = np.array(bedfile['positions'])
    return bedfile
  
                


if __name__ == '__main__':

    bwfile = pyBigWig.open(sys.argv[1])
    bedfile = readbed(sys.argv[2])

    outname = os.path.splitext(sys.argv[2])[0]+'_in_'+os.path.split(sys.argv[1])[1]

    #open bp array for each region in bedfile
    bedlen = [x[1] - x[0] for x in bedfile['positions']]
    seqlen = np.amax(bedlen)

    if '--extendbed' in sys.argv:
        exten = sys.argv[sys.argv.index('--extendbed')+1]
        outname += 'sl'+exten
        extend = int((int(exten) - seqlen) /2)
        print('Extend sequence by 2X', extend)
        seqlen = 2*extend + seqlen
        bedfile['positions'][:,0] -= extend
        bedfile['positions'][:,1] += extend
        bedfile['positions'][:,0][bedfile['positions'][:,0]<0] = 0
            


    if '--outdir' in sys.argv:
        outname = sys.argv[sys.argv.index('--outdir')+1] + os.path.split(outname)[1]


    bedcounts = np.zeros((len(bedfile['names']), seqlen), dtype = np.float32)

    print('Bed regions', len(bedfile['names']))
    i = 0
    for c, chrom in enumerate(np.unique(bedfile['contigs'])):
        # Check if chromosome exists in bamfile and get length
        if chrom in bwfile.chroms():
            print(chrom, bwfile.chroms(chrom))
            
            # get all positions in bedfile that are in chromosome
            mask = np.where(bedfile['contigs'] == chrom)[0]
            # iterate over these positions in the bedfile
            for m in mask:
                if i %10000 == 0:
                    print(i)
                i += 1
                
                bst, ben = bedfile['positions'][m]
                # fetch all reads that are within
                bwvalues = np.nan_to_num(bwfile.values(chrom, bst, ben))
                bedcounts[m] = bwvalues
                
    if '--meancounts' in sys.argv:
        mean_window= sys.argv[sys.argv.index('--meancounts')+1]
        if isint(mean_window):
            bedcounts = avgpool(bedcounts, int(mean_window))
        else:
            bedcounts = np.mean(bedcounts, axis = -1).reshape(-1,1)
        outname += 'avg'+str(mean_window)

    np.savez_compressed(outname+'.npz', names = np.array(bedfile['names']), counts = bedcounts[:,None, :])





