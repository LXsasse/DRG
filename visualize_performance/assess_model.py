import numpy as np
import sys, os
import matplotlib.pyplot as plt
from functools import reduce
import glob
import seaborn as sns
from matrix_plot import plot_heatmap, plot_distribution
from matplotlib import cm
from scipy.stats import skew




def read_in(files, reverse = False, replacenan = 1., allthesame = False, valcol= 1):
    rnames, values = [], []
    for f, fi in enumerate(files):
        print('READ', fi)
        fi = open(fi, 'r').readlines()
        names, vals = [], []
        for l, line in enumerate(fi):
            if line[0] != '#':
                line = line.strip().split()
                names.append(line[0])
                vals.append(line[valcol])
        sortnames = np.argsort(names)
        values.append(np.array(vals, dtype = float)[sortnames])
        rnames.append(np.sort(names))
    allnames = reduce(np.intersect1d, rnames)
    print('Intersect', len(allnames))
    if allthesame:
        for r, rname in enumerate(rnames):
            if len(allnames) != len(rname):
                mask = np.isin(rname, allnames)
                print(files[r], len(rname[~mask]), 'not in all sets', len(rname))
                values[r] = values[r][mask]
        values = np.array(values)
        values[np.isnan(values)] = replacenan
        if reverse:
            values = 1. - values
    else:
        for i in range(len(values)):
            values[i] = np.array(values[i])
            values[i][np.isnan(values[i])] = replacenan
            if reverse:
                values[i] = 1. - values[i]
    return allnames, values
    
def sortafter(array, target):
    array = list(array)
    sorta = []
    for t, tar in enumerate(target):
        if tar in array:
            sorta.append([array.index(tar), t])
    return np.array(sorta, dtype = int).T
        
    
files = sys.argv[1]
if '^' in files:
    files = np.sort(glob.glob(files.replace('^', '*')))
else:
    files = files.split(',')

ats = True
if '--plot_heatmap' not in sys.argv:
    ats = False
valcol = -1
if '--data_column' in sys.argv:
    valcol = int(sys.argv[sys.argv.index('--data_column')+1])
classname, classvalue = read_in(files, reverse = True, allthesame = ats, valcol = valcol)

modnames = sys.argv[2]
if modnames == 'None':
    modnames = [os.path.splitext(os.path.split(fi)[1])[0].replace('ALL.FC.PV', '').replace('-on-mouse_proteincoding-3utr-on-ALL.FC.PV_max7500_kmer-', '').replace('_pvsign', 'pvsign') for fi in files]
else:
    modnames = modnames.split(',')


means = []
for m, mn in enumerate(modnames):
    means.append([mn,np.mean(classvalue[m]), np.std(classvalue[m])])

means = np.array(means)
sortmean = np.argsort(means[:,1].astype(float))
means = means[sortmean]
classvalue = np.array(classvalue)[sortmean]
modnames = np.array(modnames)[sortmean]

if '--print_mean' in sys.argv:
    for ms in means:
        print(ms[0], ms[1], '+-', ms[2])
        
outname = None
if '--savefig' in sys.argv:
    outname = sys.argv[sys.argv.index('--savefig')+1]

swarm = True
if '--noswarm' in sys.argv:
    swarm = False
        

if '--scatter_colors' in sys.argv:
    col_features = np.genfromtxt(sys.argv[sys.argv.index('--scatter_colors')+1], dtype = str)
    replacename = sys.argv[sys.argv.index('--scatter_colors')+2].split('=')
    colfeat = int(sys.argv[sys.argv.index('--scatter_colors')+3])
    colfeatnames = col_features[:,0].astype('U50')
    for e, ce in enumerate(colfeatnames):
        colfeatnames[e] = ce.replace(replacename[0], replacename[1])
    sortfeat = sortafter(colfeatnames,classname)
    if len(sortfeat[0]) == 0:
        print("Names don't match")
        sys.exit()
    col_features = col_features[sortfeat[0]][:,colfeat].astype(float)
else:
    col_features = None
    
if '--scatter_sizes' in sys.argv:
    size_features = np.genfromtxt(sys.argv[sys.argv.index('--scatter_sizes')+1], dtype = str)
    replacename = sys.argv[sys.argv.index('--scatter_sizes')+2].split('=')
    colfeat = int(sys.argv[sys.argv.index('--scatter_sizes')+3])
    sizefeatnames = size_features[:,0].astype('U50')
    for e, ce in enumerate(sizefeatnames):
        sizefeatnames[e] = ce.replace(replacename[0], replacename[1])
    sortfeat = sortafter(sizefeatnames,classname)
    if len(sortfeat[0]) == 0:
        print("Names don't match")
        sys.exit()
    size_features = size_features[sortfeat[0]][:,colfeat].astype(float)
else:
    size_features = None


if '--plot_distribution' in sys.argv:
    # sort by highest mean
    plotnames = None
    if swarm:
        plotnames = 0
        if '--add_names' in sys.argv:
            plotnames = int(sys.argv.index('--add_names')+1)
    plot_distribution(classvalue, modnames, outname = outname, swarm = swarm, plotnames = plotnames, scatter_color = col_features, scatter_size = size_features)
    
    
elif '--plot_heatmap' in sys.argv:
    # plot heatmap for all models and cell types, cluster by euklidean distance and correlation
    # show cell type and cytokin on x or y
    # have row with mean performance
    exp_features = None
    featcol = None
    if '--get_expfeatures' in sys.argv:
        exp_features = []
        for cn in classname:
            exp_features.append(cn.split('.'))
        exp_features = np.array(exp_features)
        #check if all features equal and remove column that is all equal
        clsel = []
        for ef in exp_features.T:
            clsel.append(np.sum(ef != np.unique(ef)[0]) > 0)
        exp_features = exp_features[:, clsel]
        featcol = cm.tab20
    
    if '--scatter_sizes' in sys.argv:
        featcol = cm.jet
        if exp_features is not None:
            exp_features = np.append(exp_features, size_features.reshape(-1,1) ,axis = 1)
        else:
            exp_features = size_features.reshape(-1,1)
    
    if '--scatter_color' in sys.argv:
        featcol = cm.jet
        if exp_features is not None:
            exp_features = np.append(exp_features, col_features.reshape(-1,1) ,axis = 1)
        else:
            exp_features = col_features.reshape(-1,1)
    
    if '--load_expfeatures' in sys.argv:
        e_features = np.genfromtxt(sys.argv[sys.argv.index('--load_expfeatures')+1], dtype = str)
        replacename = sys.argv[sys.argv.index('--load_expfeatures')+2].split('=')
        colfeat = sys.argv[sys.argv.index('--load_expfeatures')+3]
        if ',' in colfeat:
            colfeat = np.array(colfeat.split(','),dtype = int)
        else:
            colfeat = [int(colfeat)]
        efeatnames = e_features[:,0].astype('U50')
        for e, ce in enumerate(efeatnames):
            efeatnames[e] = ce.replace(replacename[0], replacename[1])
        sortfeat = sortafter(efeatnames,classname)
        if len(sortfeat[0]) == 0:
            print("Names don't match")
            sys.exit()
        featcol = cm.jet
        if exp_features is not None:
            exp_features = np.append(exp_features, e_features ,axis = 1)
        else:
            exp_features = e_features
        
    
    sort = plot_heatmap(classvalue.T, measurex = 'correlation', measurey = 'euclidean', sortx = 'single', sorty = 'average', x_attributes = None, y_attributes = exp_features, xatt_color = cm.tab20, yatt_color = featcol, plot_value = True, grid = False, vmin = -np.amax(np.absolute(classvalue)), vmax = np.amax(np.absolute(classvalue)), xticklabels = modnames, yticklabels = classname, dpi = 200, figname = outname)
    
    if '--print_values' in sys.argv:
        print(modnames[sort[0]])
        for i, s in enumerate(sort[1]):
            if i%100 == 0:
                print(i)
            print(classname[s], ' '.join(np.around(classvalue.T[s][sort[0]],2).astype(str)))
    
        
    plot_distribution(classvalue[sort[0]], modnames[sort[0]], outname = outname+'_heatmap', swarm = swarm, scatter_color = col_features, scatter_size = size_features)
    
    
    

# GET number of ups and downs for each fold






