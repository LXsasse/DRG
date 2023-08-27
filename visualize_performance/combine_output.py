# combine_output.py
# combines outputs from different output files, 
# and potentially filters data points based on a list or a list of lists
# rescales outputs according to a scalar

import numpy as np
import sys, os
import glob


def string_features(string1, string2, placeholder = ['_', '-', '.'], case = False, k = 4, mink=2, ossplit = True, emphasizeto =2, emphasizelast = 2):
    if ossplit:
        string1, string2 = os.path.split(string1)[1], os.path.split(string2)[1]
    
    if case == False:
        string1, string2 = string1.upper(), string2.upper()
    if placeholder is not None:
        for p, pl in enumerate(placeholder):
            string1, string2 = string1.replace(pl, '.'), string2.replace(pl, '.')
        string1, string2 = string1.split('.'), string2.split('.')
    
    addstring = []
    for s, st in enumerate(string1):
        if (s < emphasizeto) or ((len(string1)-s) <= emphasizelast):
            addstring.append(st)
    string1 = np.append(string1, addstring)

    addstring = []
    for s, st in enumerate(string2):
        if (s < emphasizeto) or ((len(string2)-s) <= emphasizelast):
            addstring.append(st)
    string2 = np.append(string2, addstring)        
    
    if mink is None:
        ks = [k]
    else:
        ks = np.arange(mink, k+1)
    
    feats = [[],[]]
    for k in ks:
        for s, st in enumerate(string1):
            if len(st)>= k:
                for l in range(len(st)-k+1):
                    feats[0].append(st[l:l+k])
        for s, st in enumerate(string2):
            if len(st)>= k:
                for l in range(len(st)-k+1):
                    feats[1].append(st[l:l+k])
    commonfeats = np.unique(np.concatenate(feats))
    feat1, nf1 = np.unique(feats[0], return_counts = True)
    feat2, nf2 = np.unique(feats[1], return_counts = True)
    sf1, sf2 = np.zeros(len(commonfeats)), np.zeros(len(commonfeats))
    sf1[np.isin(commonfeats, feat1)] = nf1
    sf2[np.isin(commonfeats, feat2)] = nf2
    return commonfeats, sf1, sf2


strlen = np.vectorize(len)

def most_unique_feature(stringset):
    nameset = []
    for s, string in enumerate(stringset):
        weight, commons =[],[] 
        for t, tring in enumerate(stringset):
            if t != s:
                co, sf1, sf2 = string_features(string, tring)
                weight.append(sf1 - sf2)
                commons.append(co)
        comb = np.unique(np.concatenate(commons))
        wcomb = np.zeros(len(comb))
        for c, com in enumerate(commons):
            wcomb[np.isin(comb, com)] += weight[c]
        amax = np.amax(wcomb)
        mask = wcomb == amax
        best = comb[mask]
        plen=strlen(best)
        best = best[np.argmax(plen)]
        nameset.append(best)
    return nameset

def string_similarity(string1, string2, **kwargs):
    commonfeats, sf1, sf2 = string_features(string1, string2, **kwargs)
    shared = np.sum(np.amin([sf1,sf2], axis = 0))/np.sum(np.amax([sf1,sf2], axis = 0))
    return shared


def bestassign(searchset, targetset):
    sim = np.zeros((len(targetset), len(searchset)))
    for s, se1 in enumerate(targetset):
        for t, se2 in enumerate(searchset):
            sim[s,t] = string_similarity(se1, se2)
    best = -np.ones(len(targetset), dtype = int)
    '''
    sort = np.argsort(sim, axis = 1)
    for i in range(len(targetset)):
        print(targetset[i])
        for j in range(3):
            print(j, sort[i,-1-j], searchset[sort[i,-1-j]], sim[i,sort[i,-1-j]])
    sys.exit()
    '''
    order = np.argsort(sim, axis = None)[::-1]
    j = 0
    while True:
        p0 = int(order[j]/len(searchset))
        p1 = int(order[j]%len(searchset))
        if best[p0] == -1 and p1 not in best:
            best[p0] = p1
        j += 1
        if not (best == -1).any():
            break
    return best
    
    
    
    
    
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




    if '--sets' in sys.argv:
        setfiles = sys.argv[sys.argv.index('--sets')+1]
        masks = []
        if ',' in setfiles or '+' in setfiles:
            if ',' in setfiles:
                setfiles = setfiles.split(',')
            elif '+' in setfiles:
                setfiles = np.array(glob.glob(setfiles.replace('+', '*')))
                setfiles = setfiles[bestassign(setfiles, results)]
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

    if '--colors' in sys.argv:
        setfiles = sys.argv[sys.argv.index('--colors')+1]
        masks = []
        colors = []
        if ',' in setfiles or '+' in setfiles:
            if ',' in setfiles:
                setfiles = setfiles.split(',')
            elif '+' in setfiles:
                setfiles = np.array(glob.glob(setfiles.replace('+', '*')))
                setfiles = setfiles[bestassign(setfiles, results)]
            
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
                scalefiles = scalefiles[bestassign(scalefiles, results)]
        
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
                nameadds = most_unique_feature(results)
                print(nameadds)
            else:
                nameadds = np.array(nameadds.split(','))
                nameadds = nameadds[bestassign(nameadds, results)]
                
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

