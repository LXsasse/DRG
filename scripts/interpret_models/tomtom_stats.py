import numpy as np
import sys, os

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

if __name__ == '__main__':
    
    tomtom = sys.argv[1] # output tsv from tomtom
    tnames, target, pvals, qvals = readtomtom(tomtom)
    
    qvalcut = 0.05
    if '--qval' in sys.argv:
        qvalcut = float(sys.argv[sys.argv.index('--qval')+1])
    
    mask = qvals <= qvalcut
    tnames, target, pvals, qvals = tnames[mask], target[mask], pvals[mask], qvals[mask]
    
    utnames, i_ = np.unique(tnames, return_index = True)
    print('Unique filters', len(utnames))
    
    print('Unique best targets', len(np.unique(target[i_])))
    
    percent = np.percentile(qvals[i_], [5,10,50,90,95])
    print('qvalue distribution 5, 10, 50, 90, 95%', '{0:.2E} {1:.2E} {2:.2E} {3:.2E} {4:.2E}'.format(percent[0], percent[1],percent[2], percent[3], percent[4]))
    
    print('Unique targets', len(np.unique(target)))
    
    if '--tfassignment_file' in sys.argv:
        
        meme = sys.argv[sys.argv.index('--tfassignment_file')+1]
        pwms, pwmnames = read_meme(meme)
        pwmnames = np.array(pwmnames, dtype = utnames.dtype)
        hastf=np.isin(pwmnames, utnames)
        np.savetxt(os.path.splitext(tomtom)[0]+'.assign.txt', np.array([np.arange(len(pwmnames)), hastf.astype(int)]).T.astype(str), fmt = '%s')
        
        
            
            
            
        
        
