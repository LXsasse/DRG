# data_utils.py
# functions and classes to read, save, and process data

# Author: Alexander Sasse <alexander.sasse@gmail.com>

import numpy as np
import sys, os
import glob
import pandas as pd
from .sequence_utils import quick_onehot, seq_onehot
from functools import reduce


def readinlocation(regfile):
    '''
    Parameters
    ----------
    regfile: txt file
    Reads in a location file: format is as follows:
        Gene_name, region_name, location, total_length_of_gene
    Returns
    -------
    genes: np.chararray
        Unique gene names
    sequences: np.ndarray
        one-hot encoded array of regions types for all sequences
        Pads zeros to end of encoding for shorter sequences
    possible_regions:
        Names of regions, each region gets one column in the sequence array
    '''
    
    obj = np.genfromtxt(regfile, dtype = str)
    possible_regions = list(np.unique(obj[:,1]))
    genenames = obj[:,0]
    gsort = np.argsort(genenames)
    genes = list(np.unique(genenames))
    sequences = np.zeros((len(genes), np.amax(obj[:,-1].astype(int)), len(possible_regions)), dtype = np.int8)
    for l, line in enumerate(obj[gsort]):
        if ',' in line[2]:
            sequences[genes.index(line[0]), np.array(line[2].split(','), dtype = int), possible_regions.index(line[1])] =1
        else:
            sequences[genes.index(line[0]), int(line[2].split('-')[0]):int(line[2].split('-')[1]), possible_regions.index(line[1])] =1
    return np.array(genes), sequences, np.array(possible_regions)


def inputkwargs_from_string(string, definer='=', separater = '+'):
    '''
    Takes long string as input and seperates it at "separater"
    Returns dictionary with keys from before definer and values after definer
    '''
    kwargs = {}
    if definer in string:
        if separater in string:
            adjpar = string.split(separater)
        else:
            adjpar = [string]
        for p in adjpar:
            p = p.split(definer,1)
            kwargs[p[0]] = check(p[1])
    return kwargs

def add_name_from_dict(dictionary, cutkey = 2, cutitem = 3, keysep = None):
    '''
    Transforms keys and values of dictionary into a string that can be used
    as an identifier for file name
    '''
    
    addname = ''
    for key in dictionary:
        if keysep is not None:
            if keysep in str(key):
                keyd = str(key).split('_')
                apname = ''
                for ap in keyd:
                    apname += ap[0]
            else:
                apname = str(key)[:cutkey]
        else:
            apname = str(key)[:cutkey]
        
        addname += apname+str(dictionary[key])[:cutitem]
    return addname
    


def string_features(string1, string2, placeholder = ['_', '-', '.'], case = False, k = 4, mink=2, ossplit = True, emphasizeto =2, emphasizelast = 2):
    '''
    generates a feature vector for two strings
    TODO
    Clean up and make more general
    '''
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

def get_most_unique_substr(stringset):
    '''
    Returns the most unique substrings for strings in a list
    '''
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

def _compute_string_jaccard_similarity(string1, string2, **kwargs):
    '''
    Computes jaccard similarity between two strings
    '''
    commonfeats, sf1, sf2 = string_features(string1, string2, **kwargs)
    shared = np.sum(np.amin([sf1,sf2], axis = 0))/np.sum(np.amax([sf1,sf2], axis = 0))
    return shared


def return_best_matching_strings_between_sets(searchset, targetset):
    '''
    return the indices to match best match two list of strings
    '''
    sim = np.zeros((len(targetset), len(searchset)))
    for s, se1 in enumerate(targetset):
        for t, se2 in enumerate(searchset):
            sim[s,t] = _compute_string_jaccard_similarity(se1, se2)
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

def separate_sys(sysin, delimiter = ',', delimiter_1 = None):
    if delimiter in sysin:
        sysin = sysin.split(delimiter)
    else:
        sysin = [sysin]
    if delimiter_1 is not None:
        for s, sin in enumerate(sysin):
            if delimiter_1 in sin:
                sysin[s] = sin.split(delimiter_1)
            else:
                sysin[s] = [sin]
    return sysin

def find_elements_with_substring_inarray(tocheck, inset):
    '''
    return mask for tocheck if a substring of tocheck.element is a substring
    of an elemment in inset
    '''
    tocheck, inset = [t.upper() for t in tocheck], [t.upper() for t in inset]
    keep = np.zeros(len(tocheck))
    for t, tc in enumerate(tocheck):
        for i, ins in enumerate(inset):
            if tc in ins or ins in tc:
                keep[t] = 1
    return keep == 1

def get_index_from_string(string, alist, delimiter = ','):
    '''
    Give a string of names separated by ',' or other arguments to get indexes
    '''
    if delimiter in string:
        split = string.split(delimiter)
        itrack = []
        for g,gt in enumerate(split):
            gt = numbertype(gt)
            if not isinstance(gt, int):
                gt = list(alist).index(gt)
            itrack.append(gt)
        itrack = np.array(itrack)
    elif string == 'all' or string == 'complete':
        itrack = np.arange(len(alist), dtype = int)
    elif '-to-' in string:
        itrack = np.arange(int(string.split('-to-')[0]), int(string.split('-to-')[1]), dtype = int)
    elif isint(string):
        itrack = [int(string)]
    elif string in alist:
            itrack = [list(alist).index(string)]
    else:
        print(string, 'cannot be translated into indices')
        print('Select "all", "-to-", or provide list with integers or names in')
        print(alist)
        sys.exit()
    return itrack
    

def sortafter(given, target):
    sort = []
    given = list(given)
    for t, tar in enumerate(target):
        sort.append(given.index(tar))
    return np.array(sort)

def sortnames(ns):
    co = reduce(np.intersect1d, ns)
    s = []
    for n in ns:
        s.append(np.argsort(n)[np.isin(np.sort(n), co)])
    return s


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
    if len(names) == 0:
        print('Could not read tomtom file')
        sys.exit()
    names = np.array(names)
    target = np.array(target)
    pvals = np.array(pvals, dtype = float)
    qvals = np.array(qvals, dtype = float)
    return names, target, pvals, qvals

def write_tomtom(outname, datanames, pwmnames, passed, pvals, qvals, correlation, ofs, revcomp_matrix):
    obj = open(outname+'.tomtom.tsv', 'w')
    obj.write('Query_ID\tTarget_ID\tOptimal_offset\tp-value\tCorrelation\tq-value\tOrientation\n')
    for i,j in zip(passed[0], passed[1]):
        obj.write(pwmnames[i]+'\t'+datanames[j]+'\t'+str(ofs[i,j])+'\t'+str(pvals[i,j])+'\t'+str(correlation[i,j])+'\t'+str(qvals[i,j])+'\t'+str(revcomp_matrix[i,j])+'\n')
    

def readin_motif_files(pwmfile, nameline = 'Motif'):
    infmt= os.path.splitext(pwmfile)[1]    
    if infmt == '.meme':
        pwm_set, pwmnames, nts = read_meme(pwmfile)
    elif infmt == '.npz':
        pf = np.load(pwmfile, allow_pickle = True)
        pwm_set, pwmnames = pf['pwms'] , pf['pwmnames']
        nts = None
        if 'nts' in pf:
            nts = pf['nts']
    else:
        pwm_set,pwmnames,nts = read_pwm(pwmfile, nameline = nameline)
        # Returns them shape=(L,4)
    return pwm_set, pwmnames, nts

def readgenomefasta(fasta):
    fasta = gzip.open(fasta, 'rt').readlines()
    fseq = ''
    for l, line in enumerate(fasta):
        if l !=0:
            line = line.strip('\n').upper()
            fseq += line
    return fseq

def reverse_complement_seqstring(seq):
    rseq = np.array(list(seq))[::-1]
    nseq = np.copy(rseq)
    nseq[rseq == 'A'] = 'T'
    nseq[rseq == 'C'] = 'G'
    nseq[rseq == 'G'] = 'C'
    nseq[rseq == 'T'] = 'A'
    return ''.join(nseq)


def readinfasta(fastafile, minlen = 10, upper = True):
    '''
    Reads fastas for which sequence is represented as single line
    TODO: enable function to read old school fastas where sequences is devided into different lines
    '''
    
    obj = open(fastafile, 'r').readlines()
    genes = []
    sequences = []
    for l, line in enumerate(obj):
        if line[0] == '>':
            sequence = obj[l+1].strip()
            if sequence != 'Sequence unavailable' and len(sequence) > minlen:
                genes.append(line[1:].strip())
                if upper:
                    sequence = sequence.upper()
                sequences.append(sequence)
    sortgen = np.argsort(genes)
    genes, sequences = np.array(genes)[sortgen], np.array(sequences)[sortgen]
    return genes, sequences


# Combine filenames to a new output file name, removing text that is redundant in both filenames    
def create_outname(name1, name2, lword = 'on', replace_suffixes = ['.dat', '.hmot', '.txt', '.npz', '.list', '.tab', '.tsv', '.csv', '.fasta', '.fa', '.bed'], suffix = '', split_characters = ['.', '_', '-', ',']):
    nameset1 = os.path.split(name1)[1]
    for rep in replace_suffixes:
        nameset1 = nameset1.replace(rep,suffix)
    for s in split_characters:   
        nameset1 = nameset1.replace(s, "_")
    nameset1 = nameset1.split('_')
    
    
    nameset2 = os.path.split(name2)[1]
    for rep in replace_suffixes:
        nameset2 = nameset2.replace(rep,suffix)
    name2 = nameset2
    for s in split_characters:
        nameset2.replace(s, "_")
    nameset2 = nameset2.split('_')
    
    diffmask = np.ones(len(nameset1)) == 1
    for n, na1 in enumerate(nameset1):
        for m, na2 in enumerate(nameset2):
            if na1 in na2:
                diffmask[n] = False
    diff = np.array(nameset1)[diffmask]
    outname = name2
    if len(diff) > 0:
        outname+=lword+'_'.join(diff)
    return outname


# check if number can be converted to a float
def isfloat(number):
    try:
        float(number)
    except:
        return False
    else:
        return True

def isint(x):
    try:
        int(x) 
        return True
    except:
        return False

# check if string can be integer or float
def numbertype(inbool):
    try:
        int(inbool)
    except:
        pass
    else:
        return int(inbool)
    try:
        float(inbool)
    except:
        pass
    else:
        return float(inbool)
    return inbool

# check if str is boolean, a list, or a number, otherwise return string back
import ast
def check(inbool):
    if inbool == 'True' or inbool == 'TRUE' or inbool == 'true':
        return True
    elif inbool == 'False' or inbool == 'FALSE' or inbool == 'false':
        return False
    elif inbool == 'None' or inbool == 'NONE' or inbool == 'none':
        return None
    elif "[" in inbool or "(" in inbool:
        return ast.literal_eval(inbool)
    else:
        inbool = numbertype(inbool)
    return inbool


def read_matrix_file(filename, delimiter = None, name_column = 0, data_start_column = 1, value_dtype = float, header = '#', strip_names = '"', column_name_replace = None, row_name_replace = None, unknown_value = 'NA', nan_value = 0):
    '''
    Reads in text file and returns names of rows, names of colums and data matrix
    TODO
    Use pandas and switch scripts to pandas
    
    Parameters
    ----------
    filename : string
        Location of file
    delimiter : string
        Delimiter between columns
    name_column: int    
        Column in which name is placed, None means that no names are given
    data_start_column:
        Column 
    '''

    f = open(filename, 'r').readlines()
    columns, rows, values = None, [], []
    if header is not None:
        if f[0][:len(header)] == header:
            columns = f[0].strip(header).strip().replace(strip_names,'').split(delimiter)

    start = 0
    if columns is not None:
        start = 1
    
    if name_column is None:
        nc = 0
    else:
        nc = name_column
    for l, line in enumerate(f):
        if l >= start:
            line = line.strip().split(delimiter)
            rows.append(line[name_column].strip(strip_names))
            ival = np.array(line[data_start_column:])
            values.append(ival)

    if name_column is None:
        rows = np.arange(len(rows)).astype(str)
    if column_name_replace is not None:
        for c, col in enumerate(columns):
            columns[c] = col.replace(column_name_replace[0], row_name_replace[1])

    if row_name_replace is not None:
        for r, row in enumerate(rows):
            rows[r] = row.replace(row_name_replace[0], row_name_replace[1])
    try: 
        values = np.array(values, dtype = value_dtype)
    except: 
        ValueError
        values = np.array(values)
        print("matrix could not be converted to floats")
        
    if (values == np.nan).any():
        print('ATTENTION nan values in data matrix.', filename)
        if nan_value is not None:
            print('nan values replaced with', nan_value)
            values = np.nan_to_num(values, nan = nan_value)
    
    if columns is not None:
        if len(columns) > np.shape(values)[1]:
            columns = columns[-np.shape(values)[1]:]
        columns = np.array(columns)
    return np.array(rows), columns, values


def readalign_matrix_files(matrixfiles, split = ',', delimiter = None, align_rows = True, concatenate_axis = 1, align_columns = False, return_unique_names = True, **kwargs):
    
    '''
    Reads in one or several files and sorts and aligns rows to each other if 
    more than one file. Reads .npz and all sorts of txt files
    
    Parameters
    ----------
    matrixfile : String, list
        if string, individual files should be separated by 'split'
    delimiter : str 
        for txt file
    align_columns: 
        if True, columns of files will be aligned as well, and files will not be concatenated
    
    Returns
    -------
    rownames :
        Single array or list of multiple arrays
    columnnames : 
        Single array or list of multiple arrays
    Y : 
        Data matrix or data matrices
    '''
    islist = False
    if isinstance(matrixfiles,list):
        islist = True
    elif isinstance(matrixfiles, str):
        if split in matrixfiles:
            islist = True
        elif not os.path.isfile(matrixfiles):
            print('matrixfiles need to be list of files, file names separated by split, or a single file name')
            sys.exit()
        
    if not islist:
        if os.path.splitext(matrixfiles)[1] == '.npz':
            Yin = np.load(matrixfiles, allow_pickle = True)
            if 'counts' in Yin.files:
                Y = Yin['counts']
            elif 'values' in Yin.files:
                Y = Yin['values']
            rownames = Yin['names']
            columnnames = Yin['celltypes']
        else:
            rownames, columnnames, Y = read_matrix_file(matrixfiles, delimiter = delimiter, **kwargs)
    elif islist:
        Y, rownames, columnnames = [], [], []
        if not isinstance(matrixfiles, list):
            matrixfiles = matrixfiles.split(split) 
        for putfile in matrixfiles:
            if os.path.splitext(putfile)[1] == '.npz':
                Yin = np.load(putfile, allow_pickle = True)
                onames = Yin['names']
                ocolumns = Yin['celltypes']
                # need to sort to use isin for removing names taht are not shared
                sort = np.argsort(onames)
                if 'counts' in Yin.files:
                    yname = 'counts'
                elif 'values' in Yin.files:
                    yname = 'values'
                Y.append(Yin[yname][sort])
                rownames.append(onames[sort])
                columnnames.append(ocolumns)
            else:
                onames, cnames, Yin = read_matrix_file(putfile, delimiter = delimiter, **kwargs)
                sort = np.argsort(onames)
                Y.append(Yin.astype(float)[sort]) 
                rownames.append(onames[sort])
                columnnames.append(cnames)
       
        if align_rows:
            comnames = reduce(np.intersect1d, rownames)
            for i, yi in enumerate(Y):
                Y[i] = yi[np.isin(rownames[i], comnames)]
            rownames = comnames
            
        if align_columns and not None in columnnames: 
            comcolumnnames = reduce(np.intersect1d, columnnames)
            sorting = [np.argsort(coln)[np.isin(np.sort(coln), comcolumnnames)] for coln in columnnames]
            Y = [y[:, sorting[s]] for s, y in enumerate(Y)]
            columnnames = comcolumnnames
            
        if concatenate_axis == 0 and align_columns:
            rownames = np.concatenate(rownames)
            Y = np.concatenate(Y, axis = 0)
        
        if concatenate_axis == 1 and align_rows:
            columnnames = np.concatenate(columnnames)
            Y = np.concatenate(Y, axis = 1)
        
        if return_unique_names and align_rows:
            u_, sort = np.unique(rownames, return_index = True)
            if isinstance(Y, list):
                rownames, Y = rownames[sort], [y[sort] for y in Y]
            else:
                rownames, Y = rownames[sort], Y[sort]
        
    return rownames, columnnames, Y



def readin(inputfile, outputfile, delimiter = ' ', return_header = True, assign_region = True, n_features = 4, combinex = True, mirrorx = False):

    '''
    Reads inputfiles and output files for Neural network training
    aligns their data points based on names
    Can also read multiple input and output files if they are provided as string separated by ','
    '''
    
    if ',' in inputfile:
        inputfiles = inputfile.split(',')
        X = []
        inputfeatures = []
        inputnames = []
        # determines if kmerfile or sequence one-hot encoding
        arekmers = True
        for i, inputfile in enumerate(inputfiles):
            if os.path.splitext(inputfile)[1] == '.npz':
                Xin = np.load(inputfile, allow_pickle = True)
                inpnames = Xin['genenames'].astype(str)
                sortn = np.argsort(inpnames)
                inputnames.append(inpnames[sortn])
                Xi = Xin['seqfeatures']
                if len(Xi) == 2:
                    Xi, inputfeats = Xi
                else:
                    if 'featurenames' in Xin.files:
                        inputfeats = Xin['featurenames']
                    else:
                        inputfeats = np.arange(np.shape(X)[-1], dtype = int).astype(str)
                Xi = Xi[sortn]
                if len(np.shape(Xi))>2:
                    arekmers = False
                    inputfeatures.append([x+'_'+str(i) for x in inputfeats])
                else:
                    inputfeatures.append(inputfeats)
                
                
                if mirrorx and not arekmers:
                    Xi = realign(Xi)
                X.append(Xi)

            else: # For fastafiles create onehot encoding 
                inpnames, inseqs = readinfasta(inputfile)
                Xin, Xinfeatures = quick_onehot(inseqs)
                inputfeatures.append([x+'_'+str(i) for x in Xinfeatures])
                if mirrorx:
                    Xin = realign(Xin)
                X.append(Xin)
                inputnames.append(inpnames)
                arekmers = False
        # transpose, and then introduce mirrorx

        comnames = reduce(np.intersect1d, inputnames)
        X = [X[i][np.isin(inputnames[i],comnames)] for i in range(len(X))]
        if arekmers:
            inputfeatures = np.concatenate(inputfeatures)
        else:
            inputfeatures = inputfeatures[0]
            if assign_region:
                lx = len(X)
                X = [np.append(X[i], np.ones((np.shape(X[i])[0], np.shape(X[i])[1], lx), dtype = np.int8)*(np.arange(lx)==i).astype(np.int8), axis = -1) for i in range(lx)]
                inputfeatures = np.append(inputfeatures, ['F'+str(i) for i in range(lx)])
        if combinex:
            X = np.concatenate(X, axis = 1)
        inputnames = comnames
    else:
        combinex = True
        if os.path.splitext(inputfile)[1] == '.npz':
            Xin = np.load(inputfile, allow_pickle = True)
            X = Xin['seqfeatures']
            if len(X) == 2:
                X, inputfeatures = X
            else:
                if 'featurenames' in Xin.files:
                    inputfeatures = Xin['featurenames']
                else:
                    inputfeatures = np.arange(np.shape(X)[-1])
                
            arekmers = len(np.shape(X)) <= 2
            inputnames = Xin['genenames']
            if assign_region == False and not arekmers:
                inputfeatures = inputfeatures[:n_features]
                X = X[:,:,:n_features]
            if mirrorx:
                X = realign(X)
        else: # For fastafiles create onehot encoding 
            arekmers = False
            inputnames, inseqs = readinfasta(inputfile)
            X, inputfeatures = quick_onehot(inseqs)
            if mirrorx:
                X = realign(X)
    if os.path.isfile(outputfile):
        if os.path.splitext(outputfile)[1] == '.npz':
            Yin = np.load(outputfile, allow_pickle = True)
            Y, outputnames = Yin['counts'], Yin['names'] # Y should of shape (nexamples, nclasses, l_seq/n_resolution)
        else:
            Yin = np.genfromtxt(outputfile, dtype = str, delimiter = delimiter)
            Y, outputnames = Yin[:, 1:].astype(float), Yin[:,0]
        hasoutput = True
    else:
        if ',' in outputfile:
            Y, outputnames = [], []
            for putfile in outputfile.split(','):
                if os.path.splitext(putfile)[1] == '.npz':
                    Yin = np.load(putfile, allow_pickle = True)
                    onames = Yin['names']
                    sort = np.argsort(onames)
                    Y.append(Yin['counts'][sort])
                    outputnames.append(onames[sort])
                else:
                    Yin = np.genfromtxt(putfile, dtype = str, delimiter = delimiter)
                    onames = Yin[:,0]
                    sort = np.argsort(onames)
                    Y.append(Yin[:, 1:].astype(float)[sort]) 
                    outputnames.append(onames[sort])
                
            comnames = reduce(np.intersect1d, outputnames)
            for i, yi in enumerate(Y):
                Y[i] = yi[np.isin(outputnames[i], comnames)]
            outputnames = comnames
            hasoutput = True
        else:
            print(outputfile, 'not a file')
            hasoutput = False
            Y, outputnames = None, None
    #eliminate data points with no features
    if arekmers and combinex:
        Xmask = np.sum(X*X, axis = 1) > 0
        X, inputnames = X[Xmask], inputnames[Xmask]
    
    
    sortx = np.argsort(inputnames)
    if hasoutput:
        sortx = sortx[np.isin(np.sort(inputnames), outputnames)]
        sorty = np.argsort(outputnames)[np.isin(np.sort(outputnames), inputnames)]
        
    if combinex:
        X, inputnames = X[sortx], inputnames[sortx]
    else:
        X, inputnames = [x[sortx] for x in X], inputnames[sortx]
    if hasoutput:
        outputnames = outputnames[sorty]
        if isinstance(Y, list):
            Y = [y[sorty] for y in Y]
        else:
            Y = Y[sorty]
    
    if return_header and hasoutput:
        if isinstance(Y, list):
            header = []
            for p, putfile in enumerate(outputfile.split(',')):
                if os.path.splitext(putfile)[1] == '.npz':
                    Yin = np.load(putfile, allow_pickle = True)
                    if 'celltypes' in Yin.files:
                        head = Yin['celltypes']
                    else:
                        head = ['C'+str(i) for i in range(np.shape(Y[p])[1])]
                else:
                    head = open(putfile, 'r').readline()
                    if '#' in head:
                        head = head.strip('#').strip().split(delimiter)
                    else:
                        head = ['C'+str(i) for i in range(np.shape(Y[p])[1])]
                header.append(np.array(head))
                    
        else:
            if os.path.splitext(outputfile)[1] == '.npz':
                if 'celltypes' in Yin.files:
                    header = Yin['celltypes']
                else:
                    header = ['C'+str(i) for i in range(np.shape(Y)[1])]
            else:
                header = open(outputfile, 'r').readline()
                if '#' in header:
                    header = header.strip('#').strip().split(delimiter)
                else:
                    header = ['C'+str(i) for i in range(np.shape(Y)[1])]
            header = np.array(header)
    else:
        header  = None
    
    if not arekmers:
        if combinex:
            X = np.transpose(X, axes = [0,2,1])
        else:
            X = [np.transpose(x, axes = [0,2,1]) for x in X]
    if combinex:
        print('Input shapes X', np.shape(X))
    else:
        print('Input shapes X', [np.shape(x) for x in X])
            
    if isinstance(Y, list):
        print('Output shapes Y', [np.shape(y) for y in Y])
    else:
        print('Output shapes Y', np.shape(Y))
    
    return X, Y, inputnames, inputfeatures, header


def readin_sequence_return_onehot(seqfile):
    '''
    opens the npz for CNN input, or fasta file, and returns onehot encoded
    list of sequences, or np.ndarray of sequences
    '''
    if seqfile.rsplit('.',1)[-1] == 'npz':
        Xin = np.load(seqfile, allow_pickle = True)
        names = Xin['genenames'].astype(str)
        X = Xin['seqfeatures']
        if len(X) == 2:
            X, inputfeats = X
    else:
        names, seqs = readinfasta(seqfile)
        X = [seq_onehot(s) for s in seqs]
    return X, names


# TODO
# Combine read_motifs, read_meme, and read_pwm, and include in readin_motif_files

def read_motifs(pwmlist, nameline = 'Motif', delimiter = '\t', alphabet_line = 'Pos', dtype = None, info = None):
    names = []
    pwms = []
    pwm = []
    other = []
    obj = open(pwmlist, 'r').readlines()
    if dtype is None:
        dtype = pwmlist.rsplit('.',1)[-1]
    
    if dtype == 'meme':
        nameline = "MOTIF"
        delimiter = None
        alphabet_line = 'ALPHABET='
        start_motif = 0
    elif dtype == 'txt':
        nameline = "Motif"
        delimiter = '\t'
        alphabet_line = 'Pos'
        start_motif = 1
    
    for l, line in enumerate(obj):
        line = line.strip().split(delimiter)
        if ((len(line) == 0) or (line[0] == '')) and len(pwm) > 0:
            pwm = np.array(pwm, dtype = float)
            pwms.append(np.array(pwm))
            pwm = []
            names.append(name)
        elif len(line) > 0:
            if line[0] == nameline:
                name = line[1]
                pwm = []
            
            elif line[0][:len(alphabet_line)] == alphabet_line:
                nts = list(line[1:])
            
            elif isinstance(numbertype(line[start_motif]), float):
                pwm.append(line)
            
            if info is not None:
                if info in line:
                    other.append(float(line[line.index(info)+1]))
    
    if len(pwm) > 0:
        pwms.append(np.array(pwm))
        names.append(name)
    
    names = np.array(names)
    lenpwms = np.array([len(pwm) for pwm in pwms])
    if (lenpwms == len(pwms[0])).all():
        pwms = np.array(pwms)
    else:
        pwms = np.array(pwms,dtype=object)
    
    if len(other) == 0:
        return pwms, names, nts
    else:
        other = np.array(other)
        return pwms, names, other

    pwms, names = np.array(pwms, dtype = float), np.array(names)
    return pwms, names, other


# Read text files with PWMs
def read_pwm(pwmlist, nameline = 'Motif'):
    names = []
    pwms = []
    pwm = []
    obj = open(pwmlist, 'r').readlines()
    for l, line in enumerate(obj):
        line = line.strip().split('\t')
        if ((len(line) == 0) or (line[0] == '')) and len(pwm) > 0:
            pwm = np.array(pwm, dtype = float)
            pwms.append(np.array(pwm))
            pwm = []
            names.append(name)
        elif len(line) > 0:
            if line[0] == nameline:
                name = line[1]
                pwm = []
            elif line[0] == 'Pos':
                nts = line[1:]
            elif isinstance(numbertype(line[0]), int):
                pwm.append(line[1:])
    
    lenpwms = np.array([len(pwm) for pwm in pwms])
    if (lenpwms == len(pwms[0])).all():
        pwms = np.array(pwms)
    else:
        pwms = np.array(pwms,dtype=object)
    
    return pwms, np.array(names), np.array(nts)

def read_meme(pwmlist, nameline = 'MOTIF'):
    names = []
    pwms = []
    pwm = []
    obj = open(pwmlist, 'r').readlines()
    for l, line in enumerate(obj):
        line = line.strip().split()
        if ((len(line) == 0) or (line[0] == '')) and len(pwm) > 0:
            pwm = np.array(pwm, dtype = float)
            pwms.append(np.array(pwm))
            pwm = []
            names.append(name)
        elif len(line) > 0:
            if line[0] == nameline:
                name = line[1]
                pwm = []
            elif line[0] == 'ALPHABET=':
                nts = list(line[1])
            elif isinstance(numbertype(line[0]), float):
                pwm.append(line)
    if len(pwm) > 0:
        pwm = np.array(pwm, dtype = float)
        pwms.append(np.array(pwm))
        names.append(name)
    
    lenpwms = np.array([len(pwm) for pwm in pwms])
    if (lenpwms == len(pwms[0])).all():
        pwms = np.array(pwms)
    else:
        pwms = np.array(pwms,dtype=object)
    
    return pwms, np.array(names), np.array(nts)

def write_pwm(file_path, pwms, names):
    obj = open(file_path, 'w')
    for n, name in enumerate(names):
        obj.write('Motif\t'+name+'\n'+'Pos\tA\tC\tG\tT\n')
        for l, line in enumerate(pwms[n]):
            line = line.astype(float)
            obj.write(str(l+1)+'\t'+'\t'.join(np.around(line,3).astype(str))+'\n')
        obj.write('\n')
    

def rescale_pwm(pfms, infcont = False, psam = False, norm = False):
    pwms = []
    for p, pwm in enumerate(pfms):
        if infcont:
            pwm = np.log2((pwm+0.001)*float(len(pwm)))
            pwm[pwm<-2] = -2
        if psam:
            pnorm = np.amax(np.absolute(pwm), axis = 0)
            pnorm[pnorm == 0] = 1
            pwm = pwm/pnorm
        pwms.append(pwm)
    if norm:
        len_pwms = np.array([len(pwm[0]) * np.std(pwm) for pwm in pwms])
        pwms = [pwm/len_pwms[p] for p, pwm in enumerate(pwms)]
    return pwms

    
def write_meme_file(pwm, pwmname, alphabet, output_file_path, round = None, biases = None):
    """[summary]
    write the pwm to a meme file
    Args:
        pwm ([np.array]): n_filters * 4 * motif_length
        output_file_path ([type]): [description]
    """
    n_filters = len(pwm)
    print(n_filters)
    meme_file = open(output_file_path, "w")
    meme_file.write("MEME version 4 \n")
    meme_file.write("ALPHABET= "+alphabet+" \n")
    meme_file.write("strands: + -\n")

    print("Saved PWM File as : {}".format(output_file_path))

    # Switch axes if necessary
    switch = True
    for p, pw in enumerate(pwm):
        if pw.shape[1] != len(alphabet):
            switch *= False

    for i in range(0, n_filters):
        pw = pwm[i]
        if np.sum(np.absolute(np.nan_to_num(pw))) > 0:
            if round is not None:
                pw = np.around(pw,round)
            if switch:
                pw = pw.T
            meme_file.write("\n")
            meme_file.write("MOTIF %s \n" % pwmname[i])
            if biases is not None:
                meme_file.write("letter-probability matrix: alength= "+str(len(alphabet))+" w= {0} bias= {1} \n".format(np.count_nonzero(np.sum(pw, axis=0)), biases[i]))
            else:
                meme_file.write("letter-probability matrix: alength= "+str(len(alphabet))+" w= %d \n" % np.count_nonzero(np.sum(pw, axis=0)))
        for j in range(0, np.shape(pw)[-1]):
            for a in range(len(alphabet)):
                if a < len(alphabet)-1:
                    meme_file.write(str(pw[ a, j])+ "\t")
                else:
                    meme_file.write(str(pw[ a, j])+ "\n")


    meme_file.close()


def readgtf(g):
    '''
    Reads in GTF file
    TODO
        make this a pandas Dataframe
        should also pick up what kind of transcript it is, primary or secondary
        predicted or measured
    '''
    if os.path.splitext(g)[1] == '.gz':
        obj = gzip.open(g, 'rt')
    else:
        obj = open(g,'r')
    fi = obj.readlines()
    itype = []
    start, end = [], []
    chrom = []
    evidence = []
    strand = []
    gene_id, gene_type, gene_name, trans_id = [], [], [], []
    
    for l, line in enumerate(fi):
        if line[0] != '#':
            line = line.strip().split('\t')
            chrom.append(line[0])
            evidence.append(line[1]) # resource from which genes were taken, like Havana or Ensembl
            itype.append(line[2])
            start.append(int(line[3]))
            end.append(int(line[4]))
            strand.append(line[6])
            info = line[8].split(';')
            gid, gty, gna = '' ,'', ''
            for i, inf in enumerate(info):
                inf = inf.strip()
                if inf[:7] == 'gene_id':
                    inf = inf.split()
                    gid = inf[1].strip('"')
                if inf[:13] == 'transcript_id':
                    inf = inf.split()
                    gtr = inf[1].strip('"')
                if inf[:9] == 'gene_type':
                    inf = inf.split()
                    gty = inf[1].strip('"')
                if inf[:9] == 'gene_name':
                    inf = inf.split()
                    gna = inf[1].strip('"')
            gene_id.append(gid)
            gene_name.append(gna)
            gene_type.append(gty)
    return np.array([chrom, start, end, gene_name, itype, strand, gene_type, evidence, gene_id, trans_id]).T
    # before [chrom, start, end, strand, itype, gene_id, gene_type, gene_name]
