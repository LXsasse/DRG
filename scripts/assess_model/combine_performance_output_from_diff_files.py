# combine_output.py
'''
Writes list outputs (first row is name, second value) from different output
files into a single file and attaches identifier for each individual file to
the names
Can filters data points based on a list or a list of lists
Can rescale outputs according to a scalar
'''
import numpy as np
import sys, os
import glob

from drg_tools.io_utils import return_best_matching_strings_between_sets, get_most_unique_substr, string_features
    
    
    
    
    
if __name__ == '__main__':

    results = sys.argv[1]
    outname = sys.argv[2]

    if ',' in results:
        results = results.split(',')
    elif '+' in results:
        results = np.sort(glob.glob(results.replace('+', '*')))
    else:
        results = [results]

    if outname in results:
        results = np.array(results)
        results = results[results != outname]

    # read in result files
    data = []
    nan = 1
    for r, re in enumerate(results):
        print(r, 'Read', re)
        dat = np.genfromtxt(re, dtype = object)
        dat[:,0] = dat[:,0].astype(str)
        dat[:,1] = dat[:,1].astype(float)
        dat = dat[np.argsort(dat[:,0])]
        if '--similarity' in sys.argv:
            dat[:,1] = 1.-dat[:,1]
            nan = 0
        print(np.shape(dat))
        data.append(dat)



    # selected sets of genes
    if '--sets' in sys.argv:
        setfiles = sys.argv[sys.argv.index('--sets')+1]
        masks = []
        if ',' in setfiles or '+' in setfiles:
            if ',' in setfiles:
                setfiles = setfiles.split(',')
            elif '+' in setfiles:
                setfiles = np.array(glob.glob(setfiles.replace('+', '*')))
                setfiles = setfiles[return_best_matching_strings_between_sets(setfiles, results)]
            for s, da in enumerate(data):
                print(s, 'Selection', setfiles[s],'in\n', results[s])
                se = np.genfromtxt(setfiles[s], dtype = str)
                masks.append(np.isin(da[:,0], se))
        else:
            se = np.genfromtxt(setfiles, dtype = str)
            for s, da in enumerate(data):
                masks.append(np.isin(da[:,0], se))
        
        for d, da in enumerate(data):
            data[d] = da[masks[d]]
            print('Select', np.shape(data[d]))

    # assigns colors to the 
    if '--colors' in sys.argv:
        setfiles = sys.argv[sys.argv.index('--colors')+1]
        masks = []
        colors = []
        if ',' in setfiles or '+' in setfiles:
            if ',' in setfiles:
                setfiles = setfiles.split(',')
            elif '+' in setfiles:
                setfiles = np.array(glob.glob(setfiles.replace('+', '*')))
                setfiles = setfiles[return_best_matching_strings_between_sets(setfiles, results)]
            
            for s, da in enumerate(data):
                print(s, 'Selection', setfiles[s],'in\n', results[s])
                ce = np.genfromtxt(setfiles[s], dtype = object)
                ce[:,0] = ce[:,0].astype(str)
                ce[:,1] = ce[:,1].astype(str)
                se = ce[:,0]
                masks.append(np.isin(da[:,0], se))
                colors.append(ce[np.isin(se, da[:,0])])
                #print(len(colors[-1]), np.sum(masks[-1]))
        else:
            ce = np.genfromtxt(setfiles, dtype = str)
            se = ce[:,0]
            for s, da in enumerate(data):
                masks.append(np.isin(da[:,0], se))
                colors.append(ce[np.isin(se, da[:,0])])
                
        
        for d, da in enumerate(data):
            data[d] = da[masks[d]]
            print('Select', np.shape(data[d]))

    # rescale values to min max 
    if '--scale' in sys.argv:
        scalefiles = sys.argv[sys.argv.index('--scale')+1]
        scalemin = float(sys.argv[sys.argv.index('--scale')+2]) # also filters everything that is below a reproducible threshold of this
        scalemax = float(sys.argv[sys.argv.index('--scale')+3])
        masks = []
        if ',' in scalefiles or '+' in scalefiles:
            if ',' in scalefiles:
                scalefiles = scalefiles.split(',')
            elif '+' in scalefiles:
                scalefiles = np.array(glob.glob(scalefiles.replace('+', '*')))
                scalefiles = scalefiles[return_best_matching_strings_between_sets(scalefiles, results)]
        
            for s, da in enumerate(data):
                print(s, 'Scaling', scalefiles[s], 'in\n', results[s])
                se = np.genfromtxt(scalefiles[s], dtype = object)
                se[:,0] = se[:,0].astype(str)
                se[:,1] = se[:,1].astype(float)
                se = se[se[:,1].astype(float) >= scalemin]
                se = se[np.argsort(se[:,0])]
                masks.append(se[np.isin(se[:,0], da[:,0])])
        else:
            se = np.genfromtxt(scalefiles, dtype = str)
            se = se[np.argsort(se[:,0])]
            se = se[se[:,1].astype(float) >= scalemin]
            for s, da in enumerate(data):
                masks.append(se[np.isin(se[:,0], da[:,0])])
        
        for d, da in enumerate(data):
            print(np.shape(da), round(np.mean(da[:,1],axis = 0),2))
            da = da[np.isin(da[:,0], masks[d][:,0])]
            da[da[:,1]>0,1] = da[da[:,1]>0,1].astype(float)/masks[d][da[:,1]>0,1].astype(float)
            da[da[:,1]>scalemax, 1] = scalemax
            print(np.shape(da), round(np.mean(da[:,1],axis = 0),2))
            data[d] = da

    #BFO,DC8,MC,MFRP,MFPC,MZB,MO6C,NK,T4n,T8em,TGD,TREG,PDC
    ### Names that will be added to the names of the genes to identify from which file they came    
    if len(results) > 1 and not '--nonameadds' in sys.argv:
        if '--nameadds' in sys.argv:
            nameadds = sys.argv[sys.argv.index('--nameadds')+1]
            if nameadds == 'auto':
                nameadds = get_most_unique_substr(results)
                print(nameadds)
            else:
                nameadds = np.array(nameadds.split(','))
                nameadds = nameadds[return_best_matching_strings_between_sets(nameadds, results)]
                
        else:
            nameadds = [str(i) for i in range(len(results))]
        
        for d, da in enumerate(data):
            print(d, 'Adding', nameadds[d], 'for\n', results[d])
            for i, a in enumerate(da):
                da[i,0] = a[0]+'_'+nameadds[d]
            data[d] = da
            if '--colors' in sys.argv:
                color = colors[d]
                for c, a in enumerate(color):
                    color[c,0] = a[0]+'_'+nameadds[d]
                colors[d] = color

    data = np.concatenate(data, axis = 0).astype(object)
    data[:,0] = data[:,0].astype(str)
    data[:,1] = data[:,1].astype(float)
    #data[:,1] = np.around(data[:,1],4)
    np.savetxt(outname, data, fmt = '%s %1.4f')
    data[:,1] = np.nan_to_num(data[:,1].astype(float), nan = nan)
    print('\nSAVED', outname, round(np.mean(data[:,1].astype(float), axis = 0),2), '\n\n')

    if '--colors' in sys.argv:
        colors = np.concatenate(colors, axis = 0).astype(object)
        np.savetxt(os.path.splitext(outname)[0]+'_colors.txt', colors, fmt = '%s')
        print('\nSAVED', os.path.splitext(outname)[0]+'_colors.txt')

