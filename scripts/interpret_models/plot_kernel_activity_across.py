# plot_kernel_activity_across.py

import numpy as np
import sys, os
import matplotlib.pyplot as plt 
import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.patches as mpatches

from drg_tools.io_utils import read_matrix_file
from drg_tools.plotlib import plot_lines

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('matrixfile', type = str)
    parser.add_argument('filter_id')
    parser.add_argument('--split_matrix', default = False, help ='Provide string to split column name by')
    parser.add_argument('--split_column', type = int, default = -1, help ='Provide column that defines different sub matrices in matrix names')
    parser.add_argument('--split_matrix_norm', help ='If values of individual matrices should be normed to abs max of that matrix', action="store_true")
    parser.add_argument('--join_columns', type = str, default = None, help = "Either provide file that defines lineages to join individual columns by average, or provide string to split column names and use default index 0 as lineage names")
    parser.add_argument('--join_index', type = int, default = 0, help = "Provide index for lineage names for joining")
    parser.add_argument('--outname', type = str, default = None)
    parser.add_argument('--ylim', default = None, nargs=2)
    
    args = parser.parse_args()
    
    names, columns, data = read_matrix_file(args.matrixfile, data_start_column = 2)
    
    datamax = np.amax(np.abs(data), axis = 0)
    
    data = data[list(names).index(args.filter_id)]
    
    #data = pd.DataFrame(matrix, index = names, columns = columns)
    #data = data.loc(args.filter_id)
    
    data_types = None
    if args.split_matrix:
        splitnames = [i.split(args.split_matrix) for i in columns]
        matrix_type = np.array([i[args.split_column] for i in splitnames])
        newcolumns = np.array([args.split_matrix.join(np.array(i)[np.arange(len(i))-len(i)*int(args.split_column<0) != args.split_column]) for i in splitnames])
        
        data_types = np.unique(matrix_type)
        dta = []
        cols = []
        for dt in data_types:
            submatrix = data[matrix_type == dt]
            if args.split_matrix_norm:
                submatrix /= np.amax(datamax[matrix_type == dt])
            dta.append(submatrix)
            cols.append(newcolumns[matrix_type == dt])
        
        data = dta
        columns = cols
    else:
        data = [data]
        columns = [columns]
    
    if args.join_columns is not None:
        if os.path.isfile(args.join_columns):
            lineages = np.genfromtxt(args.join_columns, delimiter = '\t', dtype = str)
            data_lineages = []
            for c, col in enumerate(columns):
                d_lineages = []
                for d, cl in enumerate(col):
                    d_lineages.append(lineages[list(lineages[:,0]).index(cl), 1])
                data_lineages.append(np.array(d_lineages))
        else:
            data_lineages = []
            for c, col in enumerate(columns):
                d_lineages = []
                for d, cl in enumerate(col):
                    d_lineages.append(cl.split(args.join_columns)[args.join_index])
                data_lineages.append(np.array(d_lineages))
        
        # Reduce cell types to given cell lineages
        for c, dlin in enumerate(data_lineages):
            udlin = np.unique(dlin)
            columns[c] = udlin
            ndat = []
            for u, ud in enumerate(udlin):
                ndat.append(np.mean(data[c][dlin == ud]))
            data[c] = ndat
        
    xticklabels = np.unique(np.concatenate(columns))
    positions = []
    for col in columns:
        pos = []
        for c in col:
            pos.append(list(xticklabels).index(c))
        positions.append(pos)
    
    if args.ylim is None:
        ymax = np.amax(np.absolute(np.concatenate(data))) * 1.01
        ylim = [-ymax, ymax]
    else:
        ylim = np.array(args.ylim, dtype = float)
    
    fig = plot_lines(data, x = positions, xticks = None, xticklabels = xticklabels, color = None,
               cmap = 'Set1', marker = 's', ylabel = 'Effect', grid = True,
               legend_names = data_types, legend_outside = True, figsize = None,
               unit = 0.5, yscale = None, ylim = ylim)
    
    outname = args.outname
    if args.outname is None:
        outname = os.path.splitext(args.matrixfile)[0]
    
    fig.savefig(outname +'_'+str(args.filter_id) + '.jpg', dpi = 250, bbox_inches = 'tight')
            
            
            
        
        
        
    
    
