import sys, os 
import numpy as np
import ast

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

    

def pwmset(pwms, targetlen):
    def largerpwm(pwm, targetlen):
        npwm = np.zeros((np.shape(pwm)[0], targetlen), dtype = np.float32)
        npwm[:, int((targetlen - np.shape(pwm)[1])/2):int((targetlen - np.shape(pwm)[1])/2)+np.shape(pwm)[1]] = pwm
        return npwm
    return np.array([largerpwm(pwm,targetlen) for pwm in pwms])
        

def pwm_scan(sequences, pwms, targetlen = None, pooling_type = 'Max', pooling_size = None, pooling_steps = None, motif_cutoff = None, set_to = 0., verbose = False, maxscale = False):
    
    if targetlen is None:
        pwmlen = np.array([len(pqm.T) for pqm in pwms])
        targetlen = np.amax(pwmlen)
    scanlen = np.shape(sequences)[-1]-targetlen+1
    
    if pooling_type == 'Max':
        def combfunc(sca ,axis = -1):
            return np.amax(sca, axis = axis)
    if pooling_type == 'Mean':
        def combfunc(sca ,axis = -1):
            return np.mean(sca, axis = axis)
    
    if pooling_size is None:
        pooling_size = scanlen
        pooling_steps = scanlen
        
    steps = int((scanlen - pooling_size)/pooling_steps) + 1 + int((scanlen - pooling_size)%pooling_steps > 0)
    
    setps = pwmset(pwms, targetlen)
    outscan = np.zeros((np.shape(sequences)[0],np.shape(setps)[0], steps), dtype = np.float32)
    
    if verbose:
        print('Scanning', len(sequences), 'sequences with', len(pwms), 'PWMs with pooling', pooling_size, pooling_steps)
   
    i = 0
    s = 0
    outscanmed = np.zeros((np.shape(sequences)[0],np.shape(setps)[0], pooling_size), dtype = np.float32)
    for l in range(scanlen):
        outscanmed[:, :, i] = np.sum(sequences[:,None,:,l:l+targetlen] * setps[None, :, :, :], axis = (-1, -2))
        i += 1
        if i == pooling_size:
            if verbose:
                print(l)
            outscan[:, :, s] = combfunc(outscanmed)
            s +=1
            i -= pooling_steps
            outscanmed[:, :, :pooling_size-pooling_steps] = outscanmed[:, :, pooling_steps:]

    if maxscale:
        outscan = outscan/np.amax(outscan, axis = (0,2))[None,:, None]
    if motif_cutoff is not None:
        outscan[outscan < motif_cutoff] = set_to
    return outscan
    
    
    
def read_pwm(pwmlist, psam  = False, infcont = False):
    names = []
    pwms = []
    pwm = []
    quickmotif = []
    obj = open(pwmlist, 'r').readlines()
    for l, line in enumerate(obj):
        line = line.strip().split()
        if len(line) > 0:
            if line[0] == 'Motif':
                name = line[1]
                pwm = []
            elif line[0] == 'Pos':
                nts = line[1:]
            elif isinstance(numbertype(line[0]), int):
                pwm.append(line[1:])
        elif len(line) == 0 and len(pwm) > 0:
            pwm = np.array(pwm, dtype = float)
            if  infcont:
                pwm = np.log2((pwm+1e-8)*float(len(nts)))
                pwm[pwm<0] = 0
            if psam:
                pwm = pwm/np.amax(pwm, axis = 1)[:, None]
            pwms.append(np.array(pwm).T)
            names.append(name)
            quickmotif.append(''.join(np.array(nts)[np.argmax(pwm, axis = 1)]))
            pwm = []
            
    return pwms, names

    
def readin(inputfile):
    Xin = np.load(inputfile, allow_pickle = True)
    X, inputfeatures = Xin['seqfeatures']
    inputnames = Xin['genenames']
    X = np.transpose(X, (0,2,1))
    return X, inputnames, inputfeatures


inputfile = sys.argv[1]
X, names, features = readin(inputfile)
pwmfile = sys.argv[2]

infcont = False
if '--infocont' in sys.argv:
    infcont = True

psam = False
if '--psam' in sys.argv:
    psam = True

motcut = None
motscale = False
if '--motif_cutoff' in sys.argv:
    motscale = check(sys.argv[sys.argv.index('--motif_cutoff')+1])
    motcut = float(sys.argv[sys.argv.index('--motif_cutoff')+2])
    


targetlen = None
if '--targetlen' in sys.argv:
    targetlen = int(sys.argv[sys.argv.index('--targetlen')+1])
    
pooling_size = None
pooling_steps = None
pooling_type = 'Mean'
if '--pooling' in sys.argv:
    pooling_size = check(sys.argv[sys.argv.index('--pooling')+1])
    pooling_steps = check(sys.argv[sys.argv.index('--pooling')+2])
    pooling_type = sys.argv[sys.argv.index('--pooling')+3]


pwms, rbpnames = read_pwm(pwmfile, psam = psam, infcont = infcont)

X = pwm_scan(X, pwms, targetlen = targetlen, motif_cutoff = motcut, set_to = 0., verbose = False, maxscale = motscale, pooling_size = pooling_size, pooling_steps = pooling_steps, pooling_type = pooling_type)

if pooling_size is not None:
    nrbpnames = []
    for rbpname in rbpnames:
        for i in range(np.shape(X)[-1]):
            nrbpnames.append(rbpname+'_'+str(i*pooling_steps)+'bp')
else:
    nrbpnames = rbpnames

X = X.reshape(np.shape(X)[0], -1)
outname = os.path.splitext(inputfile)[0]+ '_on_'+ os.path.splitext(os.path.split(pwmfile)[1])[0] + '-psam'+str(psam)[0]+'-ic'+str(infcont)[0]+'_motcut'+str(motscale)[0]+str(motcut)+ '_pooling'+pooling_type+str(pooling_size)+'-'+str(pooling_steps)

print('Saved as', outname)
np.savez_compressed(outname, seqfeatures = (X, nrbpnames), genenames = names)
    
    
    
    
