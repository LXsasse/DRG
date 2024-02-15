import numpy as np
import sys, os

def readtomtom(f):
    obj = open(f,'r').readlines()
    names = []
    pvals =[]
    qvals = []
    target = []
    for l, line in enumerate(obj):
        if l > 0 and line[0] != '#':
            line = line.strip().split('\t')
            if len(line) > 5:
                names.append(line[0])
                target.append(line[1])
                pvals.append(line[3])
                qvals.append(line[5])
        
    names = np.array(names)
    target = np.array(target)
    pvals = np.array(pvals, dtype = float)
    qvals = np.array(qvals, dtype = float)
    return names, target, pvals, qvals

def findinstring(tocheck, inset):
    tocheck, inset = [t.upper() for t in tocheck], [t.upper() for t in inset]
    keep = np.zeros(len(tocheck))
    for t, tc in enumerate(tocheck):
        for i, ins in enumerate(inset):
            if tc in ins or ins in tc:
                keep[t] = 1
    return keep == 1

def filtertfset(target, targetnames, tnames, stat, tfset, tffilter, outname):
    if tffilter == 'hardexactfilter':
        outname += '_hdexftr'
        mask = np.isin(np.char.upper(target), np.char.upper(tfset)) | np.isin(np.char.upper(targetnames), np.char.upper(tfset))
        tnames, target, targetnames, stat = tnames[mask], target[mask], targetnames[mask], stat[mask]
    elif tffilter == 'softexactfilter':
        outname += '_sftexftr'
        mask = np.isin(np.char.upper(target), np.char.upper(tfset)) | np.isin(np.char.upper(targetnames), np.char.upper(tfset))
        tgnames = []
        for t, tn in targetnames:
            if not mask[t]:
                tgnames.append(tn+'<')
            else:
                tgnames.append(tn)
        targetnames = np.array(tgnames)
    elif tffilter == 'hardfilter':
        outname += '_hdftr'
        mask = findinstring(np.char.upper(targetnames), np.char.upper(tfset))
        tnames, target, targetnames, stat = tnames[mask], target[mask], targetnames[mask], stat[mask]
    elif tffilter == 'softfilter':
        outname += '_sftftr'
        mask = findinstring(np.char.upper(targetnames), np.char.upper(tfset))
        tgnames = []
        for t, tn in targetnames:
            if not mask[t]:
                tgnames.append(tn+'<')
            else:
                tgnames.append(tn)
        targetnames = np.array(tgnames)
    else:
        print('TF filter not understood')
        sys.exit()
    return target, targetnames, tnames, stat, outname

def sorttfset(target, targetnames, tnames, stat, tfmetric, tfmetfilter):
    newstat = np.zeros(len(tnames))
    addarray = np.zeros(len(tnames))
    for t in range(len(tnames)):
        #find where tnames in tfmetric
        pot = np.where(findinstring(tfmetric[:,-2], [target[t]]) | findinstring(tfmetric[:,-2], [targetnames[t]]))[0]
        if np.shape(tfmetric)[1] == 3:
            #of these preselected
            #then find where target or targetnames in tfmetric
            pot2 = np.where(tfmetric[:,0] == tnames[t])[0]
            pot = np.intersect1d(pot,pot2)
        if len(pot) > 0:    
            newstat[t] = float(tfmetric[pot[0],-1])
            addarray[t] = 1
    if tfmetfilter == 'remove':
        mask = addarray == 1
        tnames, target, targetnames, stat = tnames[mask], target[mask], targetnames[mask], stat[mask]
    else:
        tgnames = []
        for t, tn in targetnames:
            if addarray[t] ==0:
                tgnames.append(tn+'^')
            else:
                tgnames.append(tn)
        targetnames = np.array(tgnames)
    for t, tn in enumerate(np.unique(tnames)):
        mask = np.where(tnames == tn)[0]
        sort = np.lexsort(stat[mask], -newstat[mask])
        tnames[mask], target[mask], targetnames[mask], stat[mask] = tnames[mask][sort], target[mask][sort], targetnames[mask][sort], stat[mask][sort]
    return target, targetnames, tnames, stat

if __name__ == '__main__':
    
    tomtom = sys.argv[1] # output tsv from tomtom
    tnames, target, pvals, qvals = readtomtom(tomtom)
    
    vals = sys.argv[2] # detemine if pvals or qvals should be used to determine association
    if vals == 'q':
        stat = qvals
    elif vals == 'p':
        stat = pvals
    
    cut = float(sys.argv[3]) # determine cutoff for q of p-value
    
    pwms = open(sys.argv[4], 'r').readlines() # pwm file that contains the same names as 
    outname = os.path.splitext(sys.argv[4])[0] + vals + str(cut)
    
    # mask all values with too high statistics
    mask = stat <= cut
    tnames, target, stat = tnames[mask], target[mask], stat[mask]
    
    # the word characterizing the row with the pwm's name
    nameline = 'MOTIF'
    if '--nameline' in sys.argv:
        nameline = sys.argv[sys.argv.index('--nameline')+1]
    
    # split the taret names if they contain for example version names species names
    # target names is a copy of target otherwise, target can be used for filtering later
    targetnames = np.copy(target)
    if '--split_tomtomnames' in sys.argv:
        split = sys.argv[sys.argv.index('--split_tomtomnames')+1] # character used for split
        s = int(sys.argv[sys.argv.index('--split_tomtomnames')+2]) # index of entry used after split
        targetnames = np.array([tn.split(split)[s] for tn in target])
    
    # filter target names for a specific key word, f.e. MOUSE, or a specific version
    filt = None
    if '--filter_tomtomnames' in sys.argv:
        filt = sys.argv[sys.argv.index('--filter_tomtomnames')+1]
        outname += filt
    
    # if True only most significant naem will be assigned, otherwise all that pass
    only_best = False
    if '--only_best' in sys.argv:
        only_best = True
        outname += 'best'
    
    # if cluster name should be reduced from Clust_X to X for example
    rsplit = None
    if '--reduce_clustername' in sys.argv:
        rsplit = sys.argv[sys.argv.index('--reduce_clustername')+1]
    
    # filter the Transcription factor names that can be assigned to name
    if '--TFset' in sys.argv:
        tfset = sys.argv[sys.argv.index('--TFset')+1]
        outname += '.'+os.path.splitext(os.path.split(tfset)[1])[0]
        tfset = np.genfromtxt(tfset, dtype = str)
        # just need to filter entries that don't contain valid TF name, or add a marker that says that its not in the valid set. 
        tffilter = sys.argv[sys.argv.index('--TFset')+2]
        target, targetnames, tnames, stat, outname = filtertfset(target, targetnames, tnames, stat, tffilter, tfset, outname)
        
    # sort TF by new metric from high to low: this could be by correlation to activations, or expression, or fold change
    if '--TFmetric' in sys.argv:
        tfmetric = sys.argv[sys.argv.index('--TFmetric')+1]
        tfmetfilter = sys.argv[sys.argv.index('--TFmetric')+2]
        outname += '.'+os.path.splitext(os.path.split(tfmetric)[1])[0] +'.'+tfmetfilter
        tfmetric = np.genfromtxt(tfmetric, dtype = str)
        # this metric can either be for a pair or for a TF, so we need make sure that we assign it correctly and determine how deal with cases that are not in the set.
        target, targetnames, tnames, stat = sorttfset(target, targetnames, tnames, stat, tfmetric, tfmetfilter)
        
        
            
        
    
    # If selected create a file with assigned gene names instead of pwm file with differed names
    gennamefile = False
    if '--generate_namefile' in sys.argv:
        gennamefile = True
        print(outname+'_altnames.txt')
        modpwms = open(outname+'_altnames.txt', 'w')
    else:
        print(outname+os.path.splitext(sys.argv[4])[1])
        modpwms = open(outname+os.path.splitext(sys.argv[4])[1], 'w')
    
    for l, line in enumerate(pwms):
        linesplit = line.split()
        if len(linesplit) > 0:
            if linesplit[0].upper() == nameline.upper(): # check fi nameline is first string in line
                pre = line[:len(linesplit[0])+1] # use identical nameline and delimiter as in original file
                orig_name = line[len(linesplit[0])+1:].strip() # name of the cluster
                pot_name = line[len(linesplit[0])+1:].strip() # name of the cluster
                mask = tnames == pot_name # find all possible names for this cluster
                if rsplit is not None: # check if pot_name should potentially be trimmed
                    pot_name = pot_name.split(rsplit)[-1]
                # check for any matches
                if np.sum(mask) > 0:
                    ptarget, ptarname = target[mask], targetnames[mask]
                    # see if we have to filter these names for a string
                    if filt is not None:
                        keep = [filt in pt for pt in  ptarget]
                        if np.sum(keep) > 0:
                            ptarname, ptarget = ptarname[keep], ptarget[keep]
                        else:
                            ptarname = np.array([ptarname[0]+'*']) # * means that we couldn't find any with the filter but used the best name that did not have this filter
                    if only_best and len(ptarname) > 0: # only keep the most promising result in the name
                        ptarname = ptarname[[0]]
                    pot_name = pot_name + '('+','.join(ptarname)+')'
                if gennamefile: # write a file that maps the original name to the new name
                    nline = str(orig_name)+'\t'+pot_name+ '\n'
                else: # combine all the pieces to new name for pwm file
                    nline = pre +pot_name + '\n'
                modpwms.write(nline)
                
            elif gennamefile == False: # if not a name line, and we didn't select to just create a name file, just write what was present in origional pwm file
                modpwms.write(line)
        elif not gennamefile:
            modpwms.write('\n')


