# combine_lists.py 
import numpy as np
import sys, os
from drg_tools.io_utils import create_outname
from functools import reduce 
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('lists', type = str, nargs='+')
    parser.add_argument('--combine', default = 'union', help = 'union or intersect of these sets')
    parser.add_argument('--outname', default = None)
    args=parser.parse_args()
    
    lists = []
    for l, li in enumerate(args.lists):
        lists.append(np.genfromtxt(li ,dtype = str))
        print(li, len(lists[-1]))
        
    if args.combine == 'union':
        lists = reduce(np.union1d, lists)
    elif args.combine == 'intersect':
        lists = reduce(np.intersect1d, lists)
    else: 
        print('--combine union or intersect')
        sys.exit()
    
    if args.outname is None:   
        outnames = args.lists
        outname = outnames[0]
        for inp in outnames[1:]:
            outname = create_outname(inp, outname, lword = '_')
        args.outname = outname
    
    print('Left', args.outname, len(lists))
    np.savetxt(args.outname+'.'+args.combine+'.list.txt', lists, fmt = '%s')
        
        
    
    
