import numpy as np
import sys, os 
from scipy.stats import skew

'''
computes statistics for data matrix
TODO 
Enable selection of matrix axis 
'''

import argparse
from drg_tools.io_utils import readalign_matrix_files, create_outname

def compare(arr, cut, ctype):
    if ctype == 'lt':
        return arr < cut 
    if ctype == 'gt':
        return arr > cut 
    if ctype == 'leq':
        return arr <= cut 
    if ctype == 'geq':
        return arr >= cut 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('inmatrix', type = str, help = 'File name or list of files split by ","')
    #parser.add_argument('inmatrix', type = str, nargs='+')
    parser.add_argument('column', type = str)
    parser.add_argument('cutoff', type = float)
    parser.add_argument('direction', type = str, help='lt, leq, gt, geq, determines if less then, less equal, greater than, or greater equal is used for cutoff: e.g. lt: inmmatrix[column] < cutoff')
    parser.add_argument('--delimiter', default = None, type = str, help = 'Delimiter for text file')
    parser.add_argument('--requirement', default = 'any', help = 'If several matrices given, any requires only one to fullfil condition, if all then all need to fullfil it.')
    parser.add_argument('--outname', default = None)
    args=parser.parse_args()
    
    names, exp, values = readalign_matrix_files(args.inmatrix, align_columns = True, delimiter = args.delimiter)
    
    if args.outname is None:
        if ',' in args.inmatrix:
            outnames = args.inmatrix.split(',')
            outname = outnames[0]
            for inp in outnames[1:]:
                outname = create_outname(inp, outname, lword = '')
        else:
            outname = os.path.splitext(args.inmatrix)[0]
        args.outname = outname + '.'+args.column+'.'+args.direction+str(args.cutoff)
    
    if isinstance(values, list) or len(np.shape(values)) > 2:
        mask = []
        for v, vals in enumerate(values):
            mask.append(compare(vals[:, list(exp).index(args.column)], args.cutoff, args.direction))
        if args.requirement == 'any':
            mask = np.sum(mask,axis = 0) > 0
        elif args.requirement == 'all':
            mask = np.sum(mask,axis = 0) > v
    else:
        mask = compare(values[:, list(exp).index(args.column)], args.cutoff, args.direction)
    
    print('Kept', np.sum(mask), 'of', len(names), 'with', args.column, args.direction, args.cutoff)
    print(args.outname)
    np.savetxt(args.outname+'.list.txt', names[mask], fmt = '%s')
    
    







