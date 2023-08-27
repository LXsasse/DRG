import numpy as np
import sys, os
import matplotlib.pyplot as plt
from functools import reduce
import glob
import seaborn as sns
from matrix_plot import plot_heatmap, plot_distribution
from matplotlib import cm
from scipy.stats import skew

from data_processing import check


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
        print(np.shape(values[-1]))
        rnames.append(np.sort(names))
    allnames = reduce(np.intersect1d, rnames)
    print('Intersect', len(allnames))
    if allthesame:
        for r, rname in enumerate(rnames):
            if not np.array_equal(allnames,rname):
                mask = np.argsort(rname)[np.isin(np.sort(rname), allnames)]
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
        allnames = rnames
        values = values
    print('Combined', len(allnames), np.shape(values))
    return allnames, values
    
def sortafter(array, target):
    array = list(array)
    sorta = []
    for t, tar in enumerate(target):
        if tar in array:
            sorta.append([array.index(tar), t])
    return np.array(sorta, dtype = int).T
        
if __name__ == '__main__':    
    files = sys.argv[1]
    if '^' in files:
        files = np.sort(glob.glob(files.replace('^', '*')))
    else:
        files = files.split(',')

    ats = True
    if '--canbediffsets' in sys.argv and '--plot_heatmap' not in sys.argv:
        ats = False
    
    valcol = -1
    if '--data_column' in sys.argv:
        valcol = int(sys.argv[sys.argv.index('--data_column')+1])
    
    reverse = False
    ylabel = 'Distance'
    if '--similarity' in sys.argv:
        reverse = True
        ylabel = 'Similarity'
    
    if '--ylabel' in sys.argv:
        ylabel = sys.argv[sys.argv.index('--ylabel')+1]
    
    classname, classvalue = read_in(files, reverse = reverse, allthesame = ats, valcol = valcol)
    
    if '--additional_data' in sys.argv:
        nadd = sys.argv[sys.argv.index('--additional_data')+1]
        if ',' in nadd:
            nadd = nadd.split(',')
        else:
            nadd = [nadd]
        for n in nadd:
            afiles = np.sort(glob.glob(n.replace('^', '*')))
            print(len(afiles))
            aclassname, aclassvalue = read_in(afiles, reverse = reverse, allthesame = ats, valcol = valcol)
            files = np.append(files, afiles)
            for f in range(len(afiles)):
                if ats:
                    mask = np.argsort(aclassname)[np.isin(np.sort(aclassname), classname)]
                    nask = np.argsort(classname)[np.isin(np.sort(classname), aclassname)]
                    classname = classname[nask]
                    classvalue = np.append(classvalue, [aclassvalue[f][mask]], axis = 0)
                else:
                    classname.append(aclassname[f])
                    classvalue.append(aclassvalue[f])
    
        
        
    if '--filter' in sys.argv:
        datapoints = np.genfromtxt(sys.argv[sys.argv.index('--filter')+1], dtype = str)
        if isinstance(classname, list):
            for r, rname in enumerate(classname):
                mask = np.isin(rname, datapoints)
                classvalue[r] = classvalue[r][mask]
        else:
            mask = np.isin(classname, datapoints)
            print('Filter', int(np.sum(mask)))
            classvalue = classvalue[: , mask]
        
    #### need to fix classname usage below if --canbediffsets is chosen
    
    if '--rescaletop' in sys.argv:
        tops = np.genfromtxt(sys.argv[sys.argv.index('--rescaletop')+1], dtype = str)
        tops = tops[np.argsort(tops[:,0])]
        ylabel += '/replica'
        if isinstance(classname, list):
            for r, rname in enumerate(classname):
                mask = np.isin(np.sort(tops[:,0]),rname)
                classvalue[r] /= tops[mask,1].astype(float)
                classvalue[r][classvalue[r] > 2] = 2
                classvalue[r][classvalue[r] < -2] = -2
        else:
            mask = np.isin(np.sort(tops[:,0]),classname)
            classvalue /= tops[mask,1].astype(float)
            classvalue[classvalue > 2] = 2
            classvalue[classvalue < -2] = -2        
        
    # use if you want to compare 2,3, n models for different conditions, n_conditions will be the number of modnames
    join = 1
    if '--join' in sys.argv:
        join = int(sys.argv[sys.argv.index('--join')+1])
    cond = int(len(files)/join)

    modnames = sys.argv[2]
    if modnames == 'None':
        modnames = [os.path.splitext(os.path.split(fi)[1])[0].replace('ALL.FC.PV', '').replace('-on-mouse_proteincoding-3utr-on-ALL.FC.PV_max7500_kmer-', '').replace('_pvsign', 'pvsign') for fi in files]
    elif modnames == 'modelx':
        modnames = ['Model'+str(fi) for fi in range(int(len(files)/join))]
    else:
        modnames = modnames.split(',')
        for m, modname in enumerate(modnames):
            print(modname)
            print(files[m])
            if join > 1:
                for j in range(1,join):
                    print(files[m+len(modnames)*j])
            
    
    
    means = []
    for m, mn in enumerate(files):
        means.append([modnames[m%cond]+' '+str(int(m/cond)),round(np.mean(classvalue[m]),3), round(np.std(classvalue[m]),3), round(np.median(classvalue[m]),3)])
    means = np.array(means, dtype = 'object')

    
    
    if '--sortmean' in sys.argv:
        sortmean = np.argsort(means[:cond,1].astype(float))
        modnames = np.array(modnames)[sortmean]
        for j in range(join):
            means[cond*j:cond*(j+1)] = means[cond*j:cond*(j+1)][sortmean]
        classvalue = [classvalue[s+j*cond] for j in range(join) for s in sortmean]
        
    if '--sortmedian' in sys.argv:
        sortmean = np.argsort(means[:cond,3].astype(float))
        modnames = np.array(modnames)[sortmean]
        for j in range(join):
            means[cond*j:cond*(j+1)] = means[cond*j:cond*(j+1)][sortmean]
        classvalue = [classvalue[s+j*cond] for j in range(join) for s in sortmean]

    if '--print_mean' in sys.argv:
        for m in range(cond):
            for j in range(join):
                print(means[m+j*cond,0], means[m+j*cond, 3], means[m+j*cond, 1], '+-', means[m+j*cond, 2])
            
    outname = None
    if '--savefig' in sys.argv:
        outname = sys.argv[sys.argv.index('--savefig')+1]
        outname, fmt = os.path.splitext(outname)
        if '.' in fmt:
            fmt = fmt.strip('.')
        else:
            fmt = 'jpg'
        
            
    
        

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
    elif '--scatter_color' in sys.argv:
        col_features = sys.argv[sys.argv.index( '--scatter_color')+1]
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
        classname, classvalue = classname[sortfeat[1]], np.array([cv[sortfeat[1]] for cv in classvalue])
    else:
        size_features = None


    if '--plot_distribution' in sys.argv:
        # sort by highest mean
        # things to define 
        #split = 1, outname = None, xwidth = 0.6, height = 4, width = 0.8, show_mean = True, showfliers = False, showcaps = True, facecolor = None, grid = True, swarm = True, plotnames = 0, datanames = None, scatter_color = None, scatter_colormap = cm.jet, scatter_alpha = 0.8, scatter_size = 0.5, connect_swarm = True, sort = 'top', sizemax = 2, sizemin = 0.25, colormin = None, colormax = None, dpi = 200, savedpi = 200, xorder = 'size', ylabel = None, fmt = 'jpg'
        params = {}
        if len(sys.argv) > sys.argv.index('--plot_distribution')+1:
            parameters = sys.argv[sys.argv.index('--plot_distribution')+1]
            if '+' in parameters and parameters[:2] != '--' and ':' in parameters or '=' in parameters:
                parameters = parameters.split('+')
        
                for p in parameters:
                    if ':' in p and '=' in p:
                        p = p.split('=',1)
                    elif ':' in p:
                        p = p.split(':',1)
                    elif '=' in p:
                        p = p.split('=',1)
                    params[p[0]] = check(p[1])

        plot_distribution(classvalue, modnames, split = join, outname = outname, scatter_color = col_features, scatter_size = size_features, ylabel = ylabel, **params)
        
        
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






