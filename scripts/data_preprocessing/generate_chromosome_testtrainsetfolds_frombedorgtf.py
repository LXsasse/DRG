'''
Uses bed file or gtf file to create k fold sets of data points from distinct
chromosomes, so that fold sizes are close
'''

import numpy as np
import sys, os
import gzip

from drg_tools.data_processing import generatetesttrain 



if __name__ == '__main__':

    chrfile = sys.argv[1]
    if os.path.splitext(chrfile)[1] == '.bed': 
        bed = np.genfromtxt(chrfile, dtype = str)
        names = bed[:,3]
    else:
        bed = readgtf(chrfile)
        
        mask =bed[:,4] == 'gene'
        bed = bed[mask]
        
        if '--genetype' in sys.argv:
            genetype = sys.argv[sys.argv.index('--genetype')+1]
            mask =bed[:,6] == genetype
            bed = bed[mask]
            outname += '.'+genetype
        print(np.shape(bed))
        
        if '--usegeneid' in sys.argv:
            names = bed[:,-3]
        elif '--usegeneid_noversion':
            vnames = bed[:,-3]
            names = []
            for v, vn in enumerate(vnames):
                if '.' in vn:
                    vn = vn.split('.')[0]
                names.append(vn)
            names = np.array(names)
        else:
            names = bed[:,-1]
        
    outname = os.path.splitext(sys.argv[1].replace('.gz', ''))[0]+'_tset10.txt'

    if '--exclude' in sys.argv:
        exclude = sys.argv[sys.argv.index('--exclude')+1]
        if ',' in exclude:
            exclude = exclude.split(',')
        else:
            exclude = [exclude]
        mask = ~np.isin(bed[:,0], exclude)
        bed = bed[mask]
        names = names[mask]
            

    generatetesttrain(names, bed[:,0], outname, kfold = 10)


