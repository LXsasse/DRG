import numpy as np
import sys, os
import matplotlib as mpl
import logomaker
import pandas as pd
import matplotlib.pyplot as plt
from drg_tools.motif_analysis import align_compute_similarity_motifs, reverse, combine_pwms
from drg_tools.io_utils import numbertype, isint, read_pwm, read_meme
from drg_tools.plotlib import plot_pwms


    
    
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
    
    log = False
    if '--infocont' in sys.argv:
        outname += 'ic'
        log = True
    
    for p in select:
        name = pwmnames[p]
        
        pwm = pwm_set[p]
        
        
        
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
            correlation, log_pvalues, offsets, revcomp_matrix= align_compute_similarity_motifs(pwm[1:], pwm[1:], one_half = True, fill_logp_self = 1000, min_sim = np.amin(pwm_len), infocont = log, reverse_complement = np.ones(len(pwm)-1, dtype = int))
            combpwm = combine_pwms(np.array(pwm[1:], dtype = object), np.zeros(len(pwm)-1, dtype = int), log_pvalues, offsets, revcomp_matrix)
            pwm[0] = combpwm[0]
            pwm = pwm[:11]
        if '--replace_name_with_id' in sys.argv:
            name = str(p)
        fig = plot_pwms(pwm, log = log, showaxes = True)
        fig.savefig(outname+'.'+name+outadd+'.jpg', bbox_inches = 'tight', dpi = 300)
        if '--save_reversecomplement' in sys.argv:
            figr = plot_pwm([reverse(pw) for pw in pwm], log = log, showaxes = True)
            figr.savefig(outname+'.'+name+outadd+'_r.jpg', bbox_inches = 'tight', dpi = 300)
        print(outname+'.'+name+'.jpg')
        plt.close()
        #plt.show()


    
