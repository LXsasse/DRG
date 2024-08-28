import numpy as np
import sys, os

intend = '.sortedByCoord.Introns.counts.txt'
exend =  '.sortedByCoord.consExons.counts.txt'

info = np.array(open('ImmGen_CGC_sample_info.txt','r').readlines())

obj = open('Meandesign.txt', 'w')
obj.write('Samples\tFile\tReadTypei\tBatch\tCondition\n')

celltypes = []
for l, line in enumerate(info):
    line = line.strip().split()
    celltypes.append(line[-1])
    obj.write(line[0].replace('#', '_rep') + '\t'+line[1]+intend+'\t'+'intronic\t'+line[2]+'\t'+line[3].replace('PBS', 'CTRL')+'_'+line[4]+'\n')
    obj.write(line[0].replace('#', '_rep') + '\t'+line[1]+exend+'\t'+'exonic\t'+line[2]+'\t'+line[3].replace('PBS', 'CTRL')+'_'+line[4]+'\n')

obj.close()

celltypes = np.array(celltypes)
cells = np.unique(celltypes)
for c, cell in enumerate(cells):
    mask = celltypes == cell
    obj = open(cell+'design.txt', 'w')
    obj.write('Samples\tFile\tReadTypei\tBatch\tCondition\n')
    for l, line in enumerate(info[mask]):
        line = line.strip().split()
        obj.write(line[0].replace('#', '_rep') + '\t'+line[1]+intend+'\t'+'intronic\t'+line[2]+'\t'+line[3].replace('PBS', 'CTRL')+'_'+line[4]+'\n')     
        obj.write(line[0].replace('#', '_rep') + '\t'+line[1]+exend+'\t'+'exonic\t'+line[2]+'\t'+line[3].replace('PBS', 'CTRL')+'_'+line[4]+'\n')

    obj.close()


