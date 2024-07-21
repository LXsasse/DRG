# data_utils.py
# functions and classes to read, save, and process data

# Author: Alexander Sasse <alexander.sasse@gmail.com>

import numpy as np
import sys, os
import glob

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
def create_outname(name1, name2, lword = 'on'):
    name1s = os.path.split(name1)[1].replace('.dat','').replace('.hmot','').replace('.txt','').replace('.npz','').replace('.list','').replace('.csv','').replace('.tsv','').replace('.tab','').replace('-', "_").replace('.fasta','').replace('.fa','').replace('.','_').split('_')
    name2s = name2.replace('-', "_").replace('.','_').split('_')
    diffmask = np.ones(len(name1s)) == 1
    for n, na1 in enumerate(name1s):
        for m, na2 in enumerate(name2s):
            if na1 in na2:
                diffmask[n] = False
    diff = np.array(name1s)[diffmask]
    outname = os.path.split(name2)[1].replace('.dat','').replace('.hmot','').replace('.txt','').replace('.npz','').replace('.list','').replace('.csv','').replace('.tsv','').replace('.tab','').replace('.fasta','').replace('.fa','')
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


def read_txt_files(filename, delimiter = ',', header = '#', strip_names = '"', column_name_replace = None, row_name_replace = None, unknown_value = 'NA', nan_value = 0):

    f = open(filename, 'r').readlines()
    columns, rows, values = None, [], []
    if header is not None:
        if f[0][:len(header)] == header:
            columns = line.strip(header).strip().replace(strip_names,'').split(delimiter)

    start = 0
    if columns is not None:
        start = 1

    for l, line in enumerate(f):
        if l >= start:
            line = line.strip().split(delimiter)
            rows.append(line[0].strip(strip_names))
            ival = np.array(line[1:])
            ival[ival == unknown_value] = 'nan'
            values.append(ival)

    if column_name_replace is not None:
        for c, col in enumerate(columns):
            columns[c] = col.replace(column_name_replace[0], row_name_replace[1])

    if row_name_replace is not None:
        for r, row in enumerate(rows):
            rows[r] = row.replace(row_name_replace[0], row_name_replace[1])

    values = np.array(values, dtype = float)
    if (values == np.nan).any():
        print('ATTENTION nan values in data matrix.', filename)
        if nan_value is not None:
            print('nan values replaced with', nan_value)
            values = np.nan_to_num(values, nan = nan_value)
    return np.array(rows), np.array(columns), np.array(values, dtype = float)




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
    return pwms, names

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
    return pwms, names

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

    
def write_meme_file(pwm, pwmname, alphabet, output_file_path, biases = None):
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

    for i in range(0, n_filters):
        if np.sum(np.absolute(pwm[i])) > 0:
            meme_file.write("\n")
            meme_file.write("MOTIF %s \n" % pwmname[i])
            if biases is not None:
                meme_file.write("letter-probability matrix: alength= "+str(len(alphabet))+" w= {0} bias= {1} \n".format(np.count_nonzero(np.sum(pwm[i], axis=0)), biases[i]))
            else:
                meme_file.write("letter-probability matrix: alength= "+str(len(alphabet))+" w= %d \n" % np.count_nonzero(np.sum(pwm[i], axis=0)))
        for j in range(0, np.shape(pwm[i])[-1]):
            #if np.sum(pwm[i][:, j]) > 0:
                for a in range(len(alphabet)):
                    if a < len(alphabet)-1:
                        meme_file.write(str(pwm[i][ a, j])+ "\t")
                    else:
                        meme_file.write(str(pwm[i][ a, j])+ "\n")

    meme_file.close()

