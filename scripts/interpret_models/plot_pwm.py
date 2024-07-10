import numpy as np
import sys, os
import matplotlib as mpl
import logomaker
import pandas as pd
import matplotlib.pyplot as plt
from cluster_pwms import compare_ppms, reverse, combine_pwms



def plot_pwm(pwm, log = False, axes = False):
    
    if isinstance(pwm, list):
        ifcont = True
        min_sim = 5
        for pw in pwm:
            min_sim = min(min_sim, np.shape(pw)[0])
            if (pw<0).any():
                ifcont = False
        correlation, log_pvalues, offsets, revcomp_matrix, bestmatch, ctrl_ = compare_ppms(pwm, pwm, one_half = True, fill_logp_self = 1000, min_sim = min_sim, infocont = ifcont, reverse_complement = np.ones(len(pwm), dtype = int))
        pwm_len=np.array([len(pw) for pw in pwm])
        offsets = offsets[:,0]
        offleft = abs(min(0,np.amin(offsets)))
        offright = max(0,np.amax(offsets + pwm_len-np.shape(pwm[0])[0]))
        revcomp_matrix = revcomp_matrix[:,0]
        fig = plt.figure(figsize = (2.6,1*len(pwm)), dpi = 50)
        nshape = list(np.shape(pwm[0]))
        nshape[0] = nshape[0] + offleft + offright
        for p, pw in enumerate(pwm):
            ax = fig.add_subplot(len(pwm), 1, p + 1)
            if revcomp_matrix[p] == 1:
                pw = reverse(pw)
            pw0 = np.zeros(nshape)
            pw0[offleft + offsets[p]: len(pw) + offleft + offsets[p]] = pw
            pw = pw0
            lim = [min(0, -np.ceil(np.amax(-np.sum(np.ma.masked_array(pw, pw >0),axis = 1)))), np.ceil(np.amax(np.sum(np.ma.masked_array(pw, pw <0),axis = 1)))]
            if log:
                pw = np.log2((pw+1e-16)/0.25)
                pw[pw<0] = 0
                lim = [0,2]
            logomaker.Logo(pd.DataFrame(pw, columns = list('ACGT')), ax = ax, color_scheme = 'classic')
            ax.set_ylim(lim)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if not axes:
                ax.spines['left'].set_visible(False)
                ax.tick_params(labelleft = False, left = False, labelbottom = False, bottom = False)
            ax.set_yticks(lim)
    else:
        fig = plt.figure(figsize = (2.5,1), dpi = 300)
        ax = fig.add_subplot(111)
        lim = [min(0, -np.ceil(np.amax(-np.sum(np.ma.masked_array(pwm, pwm >0),axis = 1)))), np.ceil(np.amax(np.sum(np.ma.masked_array(pwm, pwm <0),axis = 1)))]
        if log:
            pwm = np.log2((pwm+1e-16)/0.25)
            pwm[pwm<0] = 0
            lim = [0,2]
        logomaker.Logo(pd.DataFrame(pwm, columns = list('ACGT')), ax = ax, color_scheme = 'classic')
        ax.set_ylim(lim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if not axes:
            ax.spines['left'].set_visible(False)
            ax.tick_params(labelleft = False, left = False, labelbottom = False, bottom = False)
        ax.set_yticks(lim)
    return fig

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

def isint(x):
    try:
        int(x)
        return True
    except:
        return False

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
    
    
if __name__ == '__main__':  
    pwmfile = sys.argv[1]
    infmt = os.path.splitext(pwmfile)[1]
    outname = os.path.splitext(pwmfile)[0]
    
    nameline = 'Motif'
    if '--nameline' in sys.argv:
        nameline = sys.argv[sys.argv.index('--nameline')+1]
    
    if infmt == '.meme':
        pwm_set,pwmnames = read_meme(pwmfile)
    else:
        pwm_set,pwmnames = read_pwm(pwmfile, nameline = nameline)

    if '--select' in sys.argv:
        select = sys.argv[sys.argv.index('--select')+1]
        if ',' in select:
            select = select.split(',')
        else:
            select = [select]
        for s, si in enumerate(select):
            if isint(si):
                select[s] = int(si)
            else:
                select[s] = list(pwmnames).index(si)
        
        
    bpwmnames = None
    if '--basepwms' in sys.argv:
        bpwm_set, bpwmnames = read_meme(sys.argv[sys.argv.index('--basepwms')+1])
    if '--clusterfile' in sys.argv:
        bpwm_cluster = np.genfromtxt(sys.argv[sys.argv.index('--clusterfile')+1], dtype = str)[:,1]
        if '--recluster' in sys.argv:
            collect = 100
        else:
            collect = 10
    
    for p in select:
        name = pwmnames[p]
        pwm = pwm_set[p]
        log = False
        if '--infocont' in sys.argv:
            outname += 'ic'
            log = True
        
        outadd = ''
        if '--basepwms' in sys.argv:
            pwm = [pwm]
            print('C'+str(p))
            outadd = 'wset'
            if ';' in name:
                filters = name.split(';')
                for f in filters[:10]:
                    print(f)
                    pwm.append(bpwm_set[list(bpwmnames).index(f)])
            
            elif '--clusterfile' in sys.argv:
                the_clust = name.split('_')[-1]
                where = np.random.permutation(np.where(bpwm_cluster == the_clust)[0])
                print(len(where))
                for f in where[:collect]:
                    pwm.append(bpwm_set[f])
            name = 'C'+str(p)
        
        if '--recluster' in sys.argv and '--clusterfile' in sys.argv:
            pwm_len = np.array([len(pw) for pw in pwm])
            correlation, log_pvalues, offsets, revcomp_matrix, bestmatch, ctrl_ = compare_ppms(pwm[1:], pwm[1:], one_half = True, fill_logp_self = 1000, min_sim = np.amin(pwm_len), infocont = log, reverse_complement = np.ones(len(pwm)-1, dtype = int))
            combpwm = combine_pwms(np.array(pwm[1:], dtype = object), np.zeros(len(pwm)-1, dtype = int), log_pvalues, offsets, revcomp_matrix)
            pwm[0] = combpwm[0]
            pwm = pwm[:11]
            
        fig = plot_pwm(pwm, log = log, axes = True)
        fig.savefig(outname+'.'+name+outadd+'.jpg', bbox_inches = 'tight', dpi = 300)
        if '--save_reversecomplement' in sys.argv:
            figr = plot_pwm([reverse(pw) for pw in pwm], log = log, axes = True)
            figr.savefig(outname+'.'+name+outadd+'_r.jpg', bbox_inches = 'tight', dpi = 300)
        print(outname+'.'+name+'.jpg')
        plt.show()


    
