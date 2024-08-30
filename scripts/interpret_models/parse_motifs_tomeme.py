'''
Reads motif files and enables manipulation of motifs, e.g. normalization
then write output in meme file. 
'''

import numpy as np
import sys, os
from drg_tools.io_utils import numbertype, readin_motif_files, write_meme_file, check


def ExpPWM(pwms):
    for p,pwm in enumerate(pwms):
        pwms[p] = np.exp(pwm)

def stripPWM(pwms, cut, metric = 'sum', relative=True):
    for p,pwm in enumerate(pwms):
        if metric == 'sum':
            pwmsum = np.sum(pwm,axis=0)
        if metric == 'absum':
            pwmsum = np.sum(np.abs(pwm),axis=0)
        if metric == 'max':
            pwmsum = np.amax(pwm,axis=0)
        if metric == 'absmax':
            pwmsum = np.amax(np.abs(pwm),axis=0)
        if relative:
            mask = np.where(pwmsum >= cut*np.amax(pwmsum))[0]
        else:
            mask = np.where(pwmsum >= cut)[0]

        if mask[-1]-1 > mask[0]:
            pwms[p] = pwm[:,mask[0]:mask[-1]+1]
        else:
            print('No entries in pwm', names[p], pwmsum)
            print('Change stripcut')
            sys.exit()

def normPWM(pwms, norm = 'sumpos'):
    for p,pwm in enumerate(pwms):
        if norm == 'sumpos':
            pwms[p] = pwm/np.sum(pwm,axis =0)
        if norm == 'absumpos':
            pwms[p] = pwm/np.sum(np.abs(pwm),axis =0)
        if norm == 'sum':
            pwms[p] = pwm/np.sum(pwm)
        if norm == 'absum':
            pwms[p] = pwm/np.sum(np.abs(pwm))
        if norm == 'maxpos':
            pwms[p] = pwm/np.amax(pwm,axis =0)
        if norm == 'absmaxpos':
            pwms[p] = pwm/np.amax(np.abs(pwm),axis =0)
        if norm == 'max':
            pwms[p] = pwm/np.amax(pwm)
        if norm == 'absmax':
            pwms[p] = pwm/np.amax(np.abs(pwm))
        if norm == 'zscore':
            pwms[p] = pwm/np.sqrt(np.sum(pwm**2))

    
if __name__ == '__main__':
    
    pwmfile = sys.argv[1]
    pwms, names, nts = readin_motif_files(pwmfile)
    outname = os.path.splitext(pwmfile)[0]
    
    pwms = [pwm.T for pwm in pwms]
    
    if '--set' in sys.argv:
        setfile = sys.argv[sys.argv.index('--set')+1]
        tset = np.genfromtxt(setfile, dtype = str)
        mask = np.where(np.isin(names, tset))[0]
        #outname += '_'+os.path.splitext(os.path.split(setfile)[1])[0]
        pwms, names = [pwms[i] for i in mask], [names[i] for i in mask]
        outname += 'sbst'+str(len(names))
        
    if '--adjust_sign' in sys.argv:
        outname += 'adjsgn'
        for p,pwm in enumerate(pwms):
            pwms[p] = np.sign(np.sum(pwm[np.argmax(np.absolute(pwm),axis = 0),np.arange(len(pwm[0]),dtype = int)]))*pwm
    
    if '--transform' in sys.argv:
        trans = sys.argv[sys.argv.index('--transform')+1]
        if ',' in trans:
            trans = trans.split(',')
        else:
            trans = [trans]
        for t, tr in enumerate(trans):
            outname += '.'+tr.replace('=','').replace('.','')
            if '=' in tr:
                tr = tr.split('=')
            else:
                tr = [tr]
            
            for i, tri in enumerate(tr):
                tr[i] = check(tri)
            if tr[0] == 'exp':
                ExpPWM(pwms)
            if tr[0] == 'norm':
                normPWM(pwms, *tr[1:])
            if tr[0] == 'strip':
                stripPWM(pwms, *tr[1:])
            
    '''
    if '--exppwms' in sys.argv:
        outname += 'exp'
        for p,pwm in enumerate(pwms):
            pwms[p] = np.exp(pwm)
    
    # only keeps entries where sum is larger than 
    if '--strip' in sys.argv:
        stripcut = float(sys.argv[sys.argv.index('--strip')+1])
        for p,pwm in enumerate(pwms):
            pwmsum = np.sum(pwm,axis=0)
            mask = np.where(pwmsum >= stripcut)[0]
            if mask[-1]-1 > mask[0]:
                pwms[p] = pwm[:,mask[0]:mask[-1]+1]
            else:
                print('No entries in pwm', names[p], pwmsum)
                print('Change stripcut')
                sys.exit()
    
    if '--norm' in sys.argv or '--normpwms' in sys.argv:
        outname += 'nrm'
        for p,pwm in enumerate(pwms):
            pwms[p] = pwm/np.sum(pwm,axis =0)
    '''
    if '--infocont' in sys.argv:
        outname += 'ic'
        for p,pwm in enumerate(pwms):
            pwm = np.log2((pwm+1e-16)/0.25)
            pwm[pwm<0] = 0
            pwms[p] = pwm
    
    if '--changenames' in sys.argv:
        clusters = np.arange(len(names)).astype(str)
    else:
        clusters = names
    outname += '.meme'
    write_meme_file(pwms, clusters, ''.join(nts), outname, round = 3)
    
    
    
    
    
