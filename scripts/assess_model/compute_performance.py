import numpy as np
from output import save_performance, print_averages, plot_scatter
from combine_output import bestassign
import sys, os
from functions import correlation, mse

def isnum(i):
    try:
        float(i)
        return True
    except: 
        return False

def read(f, delimiter=None):
    ftype = f.rsplit('.',1)[-1]
    if ftype == 'npz':
        f = np.load(f, allow_pickle = True)
        names, values, columns = f['names'], f['values'], f['columns']
    else:
        lines = open(f,'r').readlines()
        names, values, columns = [], [], None
        for l, line in enumerate(lines):
            line = line.strip('#').strip().split(delimiter)
            if l == 0 and not isnum(line[1]):
                columns = line
            else:
                names.append(line[0])
                values.append(line[1:])
        names, values = np.array(names), np.array(values, dtype = float)
        if columns is not None:
            columns = np.array(columns)
    # assign numbers if columns are not unique
    ucol, nucol = np.unique(columns, return_counts = True)
    if (nucol>1).any():
        columns = columns.astype('<U100')
        for u, uc in enumerate(ucol):
            mask = np.where(columns == uc)[0]
            for i, m in enumerate(mask):
                columns[m] = columns[m] + '_'+str(i)
    return names, values, columns


if __name__ == '__main__':
    predfile = sys.argv[1]
    measfile = sys.argv[2]
    
    outname = os.path.splitext(predfile)[0]+'_'+sys.argv[3]
    
    # Delimiter for values in output file
    delimiter = ','
    if '--delimiter' in sys.argv:
        delimiter = sys.argv[sys.argv.index('--delimiter')+1]
    
    pnames, pvals, pcolumns = read(predfile)
    mnames, mvals, mcolumns = read(measfile)
    
    # align names
    psort = np.argsort(pnames)[np.isin(np.sort(pnames), mnames)]
    msort = np.argsort(mnames)[np.isin(np.sort(mnames), pnames)]
    pnames, pvals = pnames[psort], pvals[psort]
    mnames, mvals = mnames[msort], mvals[msort]
    
    if '--select_type' in sys.argv:
        stype = sys.argv[sys.argv.index('--select_type')+1]
        mask = np.where(np.array([stype in a.split('_')[-1] for a in pcolumns]))[0]
        pvals, pcolumns = pvals[:, mask], pcolumns[mask]
    
    # align columns
    if len(pcolumns) > len(mcolumns):
        sort = bestassign(pcolumns, mcolumns)
        pcolumns, pvals = pcolumns[sort], pvals[:,sort]
        for p, pc in enumerate(pcolumns):
            print(pc, mcolumns[p])
        
    elif len(mcolumns) >= len(pcolumns):
        sort = bestassign(mcolumns, pcolumns)
        mcolumns, mvals = mcolumns[sort], mvals[:,sort]
        for p, pc in enumerate(pcolumns):
            print(pc, mcolumns[p])
    
    
    if '--select_list' in sys.argv:
        sel_list = sys.argv[sys.argv.index('--select_list')+1]
        if ',' in sel_list:
            sel_list = sel_list.split(',')
            maska = bestassign(mcolumns, sel_list)
            maskb = bestassign(pcolumns, sel_list)
        elif os.path.isfile(sel_list):
            sel_list = np.genfromtxt(sel_list, dtype = str)
            maska = bestassign(mcolumns, sel_list)
            maskb = bestassign(pcolumns, sel_list)            
        elif sel_list in mcolumns or sel_list in pcolumns:
            maska = np.where(mcolumns == sel_list)[0]
            maskb = np.where(pcolumns == sel_list)[0]
        else:
            maska = np.where(np.array([sel_list in a for a in mcolumns]))[0]
            maskb = np.where(np.array([sel_list in a for a in pcolumns]))[0]
        
        if np.array_equal(maska, maskb):
            pcolumns, mcolumns, pvals, mvals = pcolumns[maskb], mcolumns[maska], pvals[:,maskb], mvals[:,maska]
        else:
            print('Columns are not aligned')
            sys.exit()
    
    print(np.shape(pvals), np.shape(mvals))
    print(outname)
    save_performance(pvals, mvals, np.ones(len(pcolumns), dtype = int).astype(str), pcolumns, pnames, outname, sys.argv, compare_random = True)
    print_averages(pvals, mvals, np.ones(len(pcolumns), dtype = int).astype(str), sys.argv)
    
    if '--plot_examples' in sys.argv:
        if len(sys.argv) > sys.argv.index('--plot_examples') + 1:
            
            if ',' in sys.argv[sys.argv.index('--plot_examples') + 1]:
                gset = sys.argv[sys.argv.index('--plot_examples') + 1].split(',')
                outname += '_'+str(len(gset))
            elif isnum(sys.argv[sys.argv.index('--plot_examples') + 1]):
                weight = 0.75
                corrs = (1-weight)*mse(pvals, mvals, axis = 1) + weight*correlation(pvals, mvals, axis = 1)
                gset = pnames[np.argsort(corrs)[:int(sys.argv[sys.argv.index('--plot_examples') + 1])]]
                outname += '_'+str(len(gset))
            else:
                gset = [sys.argv[sys.argv.index('--plot_examples') + 1]]
                outname += '_'+ gset[0]
            mask = np.isin(pnames, gset)
            pnames, pvals, mvals = pnames[mask], pvals[mask], mvals[mask]
        outname += '.jpg'
        plot_scatter(mvals, pvals, titles = pnames, xlabel = 'Measured', ylabel = 'Predicted', dotlabel = mcolumns, indsize = 2.5, outname = outname)
        
            
            
        
        
    
