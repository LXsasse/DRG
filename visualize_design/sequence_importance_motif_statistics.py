import numpy as np
import sys, os 
import logomaker as lm 
from data_processing import readinfasta, quick_onehot, check
import pandas as pd
from generate_sequence import load_cnn_model
from modules import loss_dict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from cluster_pwms import numbertype
from plot_sequence_importance_evolution import read_pwm, saliency_map
from cluster_pwms import compare_ppms, combine_pwms
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import fisher_exact
from compare_expression_distribution import read_separated



            
def find_pwms(saliency, z_cut = 1.3, min_mot = 5, mean_sal = 0):
    salpwms = None
    # find positions in saliency that represent motifs
    if mean_sal is None:
        mean_sal = np.mean(np.sum(saliency,axis = 1))
        std_sal = np.std(np.sum(saliency,axis = 1))
    else:
        std_sal = np.sqrt(np.mean((np.sum(saliency,axis = 1)-mean_sal)**2))
    zscore_sal = (np.sum(saliency, axis =1) - mean_sal)/std_sal
    saliency = (saliency - mean_sal)/std_sal
    potmotifs = np.absolute(zscore_sal) >= z_cut
    #print(np.where(potmotifs)[0])
    zsign = np.sign(zscore_sal) * potmotifs
    lmot, lgap, ngap, lstr, ps, sig = 0, 0, 0, 0, None, None
    # detemine motifs in saliency map as positions with enriched saliency scores
    salpstart, salpend = [], []
    # acount for different signs
    # find continous stretches with same sign # allow one gap per 7 basepairs, minimum continous stretch must be 3 within and 1 one the flanks
    p = 0
    while True:
        if potmotifs[p]:
            if lmot ==0:
                ps = p
                sig = zsign[p]
                ngap = 0
                lstr = 0
            if zsign[p] == sig:
                lmot += 1
                lgap = 0
                lstr += 1
            else:
                lgap += 1
                ngap +=1
                if ngap == 2:
                    if lmot < min_mot:
                        p = p - (lstr+1)
                        lgap, ngap, lmot = 0, 0, 0
                    else:
                        salpstart.append(ps)
                        salpend.append(p-1)
                        p = p -1
                        lgap, ngap, lmot = 0, 0, 0
                lstr = 0
        else:
            lgap += 1
            ngap +=1
            if ngap == 2:
                if lmot < min_mot:
                    p = p - (lstr+1)
                    lgap, ngap, lmot = 0, 0, 0
                else:
                    salpstart.append(ps)
                    salpend.append(p - 1)
                    p = p -1
                    lgap, ngap, lmot = 0, 0, 0
            lstr = 0
            if lgap == 2:
                if lmot >= min_mot and (ngap-2)/lmot <= 1./(lmot+2.):
                        salpstart.append(ps)
                        salpend.append(p - 2)

                p = p - 2*(zsign[p-2:p] != sig).any()
                sig = None
                lmot,ngap = 0,0
        p += 1
        if p == len(potmotifs):
            break
    #print(salpstart, salpend)
    det_direction = []
    detected_pwms = []
    for s, sst in enumerate(salpstart):
        detected_pwms.append(np.absolute(saliency[sst:salpend[s]+1]))
        det_direction.append(np.sign(np.sum(saliency[sst:salpend[s]+1])))
    return detected_pwms, det_direction

def read_optseqs(seed_seqs, seed_names, evofile):
    obj = open(evofile, 'r').readlines()
    opt = []
    opt_pos = []
    opt_names = []
    for l, line in enumerate(obj):
        line = line.strip().split(';')
        name = line[0].split(',')[0]
        opt_names.append(name)
        line = line[1:]
        seedseq = seed_seqs[list(seed_names).index(name)]
        op = [seedseq]
        opp = [None]
        for e, ent in enumerate(line):
            ent = ent.split(',')
            pos = int(ent[0])
            opp.append(pos)
            orig, new = ent[1].split('/')
            seedseq = seedseq[:pos]+new+seedseq[pos+1:]
            if e == len(line)-1:
                op.append(seedseq)
        opt.append(op)
    return opt

def readmulti(target_tracks, delimiter = ',', dtype = int):
    if delimiter in target_tracks:
        target_tracks = np.array(target_tracks.split(delimiter), dtype = dtype)
    else:
        target_tracks = [dtype(target_tracks)]
    return target_tracks

def num_motifs_insequences(clusters, pwmstats):
    pwmcounts = []
    pwmsin = []
    for t in range(len(target_tracks)):
        pwmcount = []
        # unique pwm clusters in this condition
        clint = np.unique(clusters[pwmstats[:,0]==t])
        pwmsin.append(clint)
        for i in range(2):
            pwmc = []
            # look at motifs seperately if they have negative influence
            for j in [1,-1]:
                countm = np.zeros(len(clint))
                clm, cln = np.unique(clusters[(pwmstats[:,0]==t) & (pwmstats[:,1]==i) & (pwmstats[:,2] == j)], return_counts = True)
                countm[np.isin(clint,clm)] = j*cln
                pwmc.append(countm)
            
            pwmcount.append(pwmc)
        pwmcounts.append(pwmcount)
    return pwmcounts, pwmsin



if __name__ == '__main__':

    seed_names, seed_seqs = readinfasta(sys.argv[1])
    
    evofile = sys.argv[2]
    
    opt_names, opt_seqs = readinfasta(evofile)
    
    outname = os.path.splitext(evofile)[0]
    
    opt = []
    for i in range(min(len(opt_names), len(seed_names))):
        opt.append([seed_seqs[i], opt_seqs[i]])
        
    Nseqs = len(opt)
    
    predictor = sys.argv[3]

    target_tracks, tset_tracks = read_separated(sys.argv[4])
    
    ctype = sys.argv[5]
    if ',' in ctype:
        ctype = ctype.split(',')
        if len(ctype) != len(target_tracks) and tset_tracks is not None:
            tsetvalues = []
            for t, tval in enumerate(ctype):
                chara = np.chararray(len(tset_tracks[t]), itemsize = 100, unicode = True)
                chara[:] = tval
                tsetvalues.append(chara)
            ctype = np.concatenate(tsetvalues)
    else:
        ctype = [ctype]
    ctype = np.array(ctype)
    
    print(ctype)
    
    target_values = sys.argv[6]
    if ',' in target_values:
        target_values = target_values.split(',')
        if len(target_values) != len(target_tracks) and tset_tracks is not None:
            tsetvalues = []
            for t, tval in enumerate(target_values):
                tsetvalues.append(np.ones(len(tset_tracks[t]))*float(tval))
            target_values = np.concatenate(tsetvalues)
    else:
        target_values = [target_values]
    target_values = np.array(target_values, dtype = float)
    
    uctype = np.unique(ctype)
    tvalues, ttargets = [], []
    for ct in uctype:
        tvalues.append(target_values[ctype == ct])
        ttargets.append(target_tracks[ctype == ct])
    target_tracks = ttargets
    target_values = tvalues
    ctype = uctype
    
    outname += '_'+'-'.join(ctype)
    
    scoring = sys.argv[7]

    loss_function = sys.argv[8]

    outname += '_'+scoring+'_'+loss_function

    pwmfile = sys.argv[9]
    
    outname += os.path.splitext(os.path.split(pwmfile)[1])[0]
    
    pfms, rbpnames, rbpfeat = read_pwm(pwmfile, nameline = 'TF Name')
    npfms = []
    for p, pfm in enumerate(pfms):
        pfm = np.log2((pfm+1e-8)/0.25)
        pfm[pfm<0] = 0
        npfms.append(pfm)
    
    rbpnames = np.array([rname.split(';')[0] if ';' in rname else rname for rname in rbpnames])
    pfms = np.array(npfms)
    
    # make sure only pwms of tf that are expressed in the cell type are looked at
    if 'Ctype' in rbpfeat.keys():
        considerpfm = np.ones((len(rbpnames),len(target_tracks))) == 0
        for r, rbf in enumerate(rbpfeat['Ctype']):
            for t, trac in enumerate(ctype):
                if rbf !='':
                    if trac in rbf.split(','):
                        considerpfm[r, t] = True
    else:
        considerpfm = np.ones((len(rbpnames),len(target_tracks))) == 1
    # set to all TRue in not specified by Ctype
    considerpfm[:, np.sum(considerpfm, axis = 0) == 0] = True
    print('Considered PFMs', np.sum(considerpfm, axis = 0))
    
    
    
    if '--extracted_pwms' in sys.argv:
        efile = np.load(sys.argv[sys.argv.index('--extracted_pwms')+1], allow_pickle = True)
        pwmstats = efile['pwmstats']
        ext_pwms = efile['ext_pwms']
    else:
        # collect significant pwms from importat position in each sequence
        ext_pwms = []
        pwmstats = []
        for q in range(0, len(opt)):
            print('Seed',q)
            seqset = opt[q]
            seqset, nts = quick_onehot(seqset)
            seqset = np.transpose(seqset, axes=(0,2,1))
            for t, tt in enumerate(target_tracks):
                model = load_cnn_model(predictor, verbose = False)
                model.classifier.Linear.weight = nn.Parameter(model.classifier.Linear.weight[tt])
                model.classifier.Linear.bias = nn.Parameter(model.classifier.Linear.bias[tt])
                for i in range(2):
                    ohseq = seqset[[i]]
                    pwm, loss, pred = saliency_map(ohseq, target_values[t], loss_function, model, scoring)
                    #print(loss, pred, np.shape(pwm))
                    detected_pwms, direction = find_pwms(pwm.T, z_cut = 1.3, min_mot = 5, mean_sal = 0)
                    #print(detected_pwms, direction)
                    for d, dtpwm in enumerate(detected_pwms):
                        ext_pwms.append(dtpwm)
                        pwmstats.append([t, i, direction[d], q])
        pwmstats = np.array(pwmstats)
        np.savez_compressed(outname+'_extract_pwms.npz', pwmstats = pwmstats, ext_pwms =ext_pwms)
        print(outname+'_extract_pwms.npz')
        
    if '--load_clusters' in sys.argv:
        efile = np.load(sys.argv[sys.argv.index('--load_clusters')+1], allow_pickle = True)
        #clusters = efile['clusters']
        log_pvalues = efile['log_pvalues']
        offsets = efile['offsets']
        correlation = efile['correlation']
    else:
        # cluster extracted motifs
        correlation, log_pvalues, offsets, bestmatch = compare_ppms(ext_pwms, ext_pwms, find_bestmatch = True, fill_logp_self = 0, one_half = True, min_sim = 6, padding = 0., infocont = False, bk_freq = 0.25, non_zero_elements = False)
        np.savez_compressed(outname+'_extract_pwm_clusters.npz', log_pvalues = log_pvalues, offsets = offsets, correlation = correlation)
        print(outname+'_extract_pwm_clusters.npz')
        
    linkage = 'complete'
    distance_threshold = 0.3
    
    clustering = AgglomerativeClustering(n_clusters = None, affinity = 'precomputed', linkage = linkage, distance_threshold = distance_threshold).fit(correlation)
    
    clusters = clustering.labels_
    unclusters, unclustnum = np.unique(clusters, return_counts = True)
    '''
    for uc in unclusters:
        print('cluster', uc)
        incl = np.where(clusters == uc)[0]
        print(offsets[incl][:,incl])
        print(correlation[incl][:, incl])
        for p in incl:
            print(''.join(np.array(list('ACGT'))[np.argmax(ext_pwms[p],axis = 1)]))
    sys.exit()
    '''
    
    clusterpwms = combine_pwms(ext_pwms, clusters, log_pvalues, offsets, maxnorm =True, method = 'mean', remove_low = 0.1)
    

    if '--load_clusternames' in sys.argv:
        efile = np.load(sys.argv[sys.argv.index('--load_clusternames')+1], allow_pickle = True)
        motifnames = efile['motifnames']
        allmotifnames = efile['allmotifnames']
        
    else:
        # compare to known motifs and assign name if known
        if (np.sum(considerpfm, axis = 0) == len(considerpfm)).all():
            correlation, log_pvalues, offsets, bestmatch = compare_ppms(clusterpwms, pfms, find_bestmatch = True, fill_logp_self = 0, one_half = False, min_sim = 5, padding = 0., infocont = False, bk_freq = 0.25, non_zero_elements = False)
            allmatches = [[np.argsort(correlation,axis = 1), np.sort(correlation,axis = 1)] for t in range(len(tartet_tracks))]
            bestmatches = [[np.argmin(correlation,axis = 1), np.amin(correlation,axis = 1)] for t in range(len(tartet_tracks))]
        else:
            bestmatches = []
            allmatches = []
            for t in range(len(target_tracks)):
                print(len(pfms[considerpfm[:,t]]), len(clusterpwms))
                correlation, log_pvalues, offsets, bestmatch = compare_ppms(clusterpwms, pfms[considerpfm[:,t]], find_bestmatch = True, fill_logp_self = 0, one_half = False, min_sim = 5, padding = 0., infocont = False, bk_freq = 0.25, non_zero_elements = False)
                print(np.shape(correlation))
                bestmatches.append([np.where(considerpfm[:,t])[0][np.argmin(correlation,axis = 1)], np.amin(correlation,axis = 1)])
                allmatches.append([np.where(considerpfm[:,t])[0][np.argsort(correlation,axis = 1)], np.sort(correlation,axis = 1)])
        
        #pvalue_cut = -np.log10(0.001)
        pvalue_cut = 0.35
        allmotifnames = [[] for t in range(len(target_tracks))]
        motifnames = [[] for t in range(len(target_tracks))]
        for t in range(len(target_tracks)):
            for i in range(len(clusterpwms)):
                if bestmatches[t][1][i] < pvalue_cut:
                    motifnames[t].append(rbpnames[bestmatches[t][0][i]]+'('+ str(i)+')')
                    urpbnames, uindex = np.unique(rbpnames[allmatches[t][0][i][allmatches[t][1][i]<pvalue_cut]], return_index = True)
                    #print(urpbnames,uindex)
                    allmotifnames[t].append('|'.join(urpbnames[np.argsort(uindex)])+'('+ str(i)+')')
                else:
                    motifnames[t].append('Cluster'+str(i))
                    allmotifnames[t].append(rbpnames[allmatches[t][0][i][0]]+'(Cluster'+str(i)+')')
                #print(allmotifnames[t][-1], motifnames[t][-1])
        np.savez_compressed(outname+'_extract_pwm_clusternames.npz', motifnames = motifnames, allmotifnames = allmotifnames)
        print(outname+'_extract_pwm_clusternames.npz')
    # for each condition make sorted barplot, sorted by number of motifs in optimized sequences, pwms on the xlabel
    # count the nmber of motifs in any sequence, in different conditions and for start and optimized sequences
    pwmcounts, pwmsin = num_motifs_insequences(clusters, pwmstats)
    
    all_pwmsin = np.unique(np.concatenate(pwmsin))
    allpwmcounts = np.zeros((len(pwmcounts),2,2,len(all_pwmsin)))
    for t, pwnin in enumerate(pwmsin):
        for i in range(2):
            for j in range(2):
                sortpwms = np.argsort(pwnin)
                allpwmcounts[t,i,j,np.isin(all_pwmsin, pwnin)] = pwmcounts[t][i][j][sortpwms]
    sortallpwm = np.argsort(-np.sum(np.absolute(allpwmcounts[:,1]), axis = (0,1)))
    all_pwmsin, allpwmcounts = all_pwmsin[...,sortallpwm], allpwmcounts[...,sortallpwm]
    
    header = 'Motifs'
    outmat = [np.array(allmotifnames)[0][all_pwmsin]]
    for c, ct in enumerate(ctype):
        for i, iw in enumerate(('end', 'start')):
            for j, jw in enumerate(('positve', 'negative')):
                header += '\t'+ct+'_'+iw+'_'+jw
                outmat.append(allpwmcounts[c,1-i,j].astype(int).astype(str))
    np.savetxt(outname + '_stats.dat', np.array(outmat).T, header = header, fmt = '%s')
    
    if '--plotpwms' in sys.argv:
        minocc = 3
        absum = np.sum(np.absolute(allpwmcounts), axis = (0,1,2))
        abmask = np.where(absum > minocc)[0]
        for m in abmask:
            pi  = all_pwmsin[m]
            l =list(unclusters).index(pi)
            fig = plt.figure(figsize = (1,0.5), dpi = 200)
            axp = fig.add_subplot(111)
            axp.spines['right'].set_visible(False)
            axp.spines['top'].set_visible(False)
            axp.spines['left'].set_visible(False)
            axp.spines['bottom'].set_visible(False)
            axp.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)
            
            cpwm = clusterpwms[l].T
            cpwmnames = motifnames[0][l]
            if motifnames[1][l] != cpwmnames:
               cpwmnames = motifnames[0][l].split('(')[0]+'-'+motifnames[1][l]
            cpwm = pd.DataFrame({'A':cpwm[0],'C':cpwm[1], 'G':cpwm[2], 'T':cpwm[3]})
            lm.Logo(cpwm, ax = axp)
            axp.set_xlabel(cpwmnames,rotation = 0)
            fig.savefig(outname + cpwmnames.replace(")",'_').replace("(",'_')+'.png', bbox_inches = 'tight', transparent = True)
            print(np.sum(allpwmcounts[:,1],axis = 1)[:,m], outname + cpwmnames.replace(")",'_').replace("(",'_')+'.png')
            plt.close()
        
    if '--plot_counts_same' in sys.argv:
        # plot number of motif and their motif in form of a pwm
        colors = ['limegreen', 'purple']
        names = ['random', 'optimized']
        
        absum = np.sum(np.absolute(allpwmcounts), axis = (0,1,2))
        abmask = absum > 2
        allpwmcounts = allpwmcounts[...,abmask]
        all_pwmsin = all_pwmsin[abmask]
            
        maxpwms = len(all_pwmsin)
        
        cols, rows = 4*len(target_tracks), maxpwms*0.6
        fig = plt.figure(figsize = (cols, rows), dpi = 100)
        motblocky = 0.8/maxpwms
        motblockx = rows/cols*2*0.8/maxpwms
        
        for t in range(len(target_tracks)):
            ax = fig.add_subplot(1,len(target_tracks),t+1)
            pwmcount = allpwmcounts[t]
            ax.set_position([0.1+motblockx*0.65 +t*(0.8-motblockx*0.65)/len(target_tracks), 0.9-len(all_pwmsin)*0.8/maxpwms, 0.9*(0.8-motblockx*0.65)/len(target_tracks), len(all_pwmsin)*0.8/maxpwms])
            for i in [1,0]:
                pwmc = pwmcount[i]
                ax.barh(-np.arange(len(pwmc[0])), pwmc[0], color = colors[i], alpha = 0.6, label = names[i])
                ax.barh(-np.arange(len(pwmc[1])), pwmc[1], alpha = 0.6, color = colors[i])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('Motif impact occurrence')
            ax.tick_params(left = False, labelleft = False)
            ax.set_ylim([-len(pwmc[0])+0.5,0.5])
            ax.plot([0,0],[-len(pwmc[0])+0.5,0.5], color = 'k')
            xlim = ax.get_position()
            ax.legend()
            ax.set_xticks([-5,-2,2,5,10,15])
            ax.grid(axis = 'x')
            ax.set_title(ctype[t]+'('+str(np.mean(target_values[t])) + ')')
        for p, pi in enumerate(all_pwmsin):
            l =list(unclusters).index(pi)
            axp = fig.add_subplot(maxpwms, 1, p+1)
            axp.set_position([0.1, xlim.y0 + xlim.height-(p+1)*0.8/maxpwms + 0.2*0.8/maxpwms, 0.6* motblockx, 0.6*motblocky])
            axp.spines['right'].set_visible(False)
            axp.spines['top'].set_visible(False)
            axp.spines['left'].set_visible(False)
            axp.spines['bottom'].set_visible(False)
            axp.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)
            cpwm = clusterpwms[l].T
            cpwmnames = motifnames[0][l]
            if motifnames[1][l] != cpwmnames:
               cpwmnames = motifnames[0][l].split('(')[0]+'-'+motifnames[1][l]
            cpwm = pd.DataFrame({'A':cpwm[0],'C':cpwm[1], 'G':cpwm[2], 'T':cpwm[3]})
            lm.Logo(cpwm, ax = axp)
            axp.set_xlabel(cpwmnames,rotation = 0, loc = 'right')
        
        fig.savefig(outname +'_motif_cluster_countssorted.jpg', dpi = 150, bbox_inches = 'tight')
        print(outname +'_motif_cluster_countssorted.jpg')
    
    if '--plot_counts' in sys.argv:
        # plot number of motif and their motif in form of a pwm
        colors = ['limegreen', 'purple']
        names = ['random', 'optimized']
        
        for t, pwnin in enumerate(pwmsin):
            absum = np.sum(np.sum(np.absolute(pwmcounts[t]),axis = 0),axis = 0)
            abmask = absum > 1
            pwmcounts[t] = np.array(pwmcounts[t])[:,:,abmask]
            pwmsin[t] = pwnin[abmask]
            
        maxpwms = np.amax([len(pwmcount) for pwmcount in pwmsin])
        
        cols, rows = 4*len(target_tracks), maxpwms*0.6
        fig = plt.figure(figsize = (cols, rows), dpi = 100)
        for t in range(len(target_tracks)):
            ax = fig.add_subplot(1,len(target_tracks),t+1)
            pwmcount = pwmcounts[t]
            ax.set_position([0.1+t*0.8/len(target_tracks), 0.9-len(pwmsin[t])*0.8/maxpwms, 0.8/len(target_tracks)- rows/cols *2 *0.8/maxpwms , len(pwmsin[t])*0.8/maxpwms])
            sortpwms = np.argsort(-pwmcount[1][0])
            for i in [1,0]:
                pwmc = pwmcount[i]
                ax.barh(-np.arange(len(pwmc[0])), pwmc[0][sortpwms], color = colors[i], alpha = 0.6, label = names[i])
                ax.barh(-np.arange(len(pwmc[1])), pwmc[1][sortpwms], alpha = 0.6, color = colors[i])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('Motif impact occurrence')
            ax.tick_params(left = False, labelleft = False)
            ax.set_ylim([-len(pwmc[0])+0.5,0.5])
            ax.plot([0,0],[-len(pwmc[0])+0.5,0.5], color = 'k')
            xlim = ax.get_position()
            ax.legend()
            ax.set_xticks([-5,-2,2,5,10,15])
            ax.grid(axis = 'x')
            ax.set_title(ctype[t]+'('+str(np.mean(target_values[t])) + ')')
            for p, pi in enumerate(pwmsin[t][sortpwms]):
                l =list(unclusters).index(pi)
                axp = fig.add_subplot(maxpwms, maxpwms, t*maxpwms+p+1)
                axp.set_position([xlim.x0-rows/cols * 2 *0.8/maxpwms + 0.2*rows/cols * 2 *0.8/maxpwms, xlim.y0 + xlim.height-(p+1)*0.8/maxpwms + 0.2*0.8/maxpwms, 0.6*rows/cols *2 *0.8/maxpwms,0.6*0.8/maxpwms])
                axp.spines['right'].set_visible(False)
                axp.spines['top'].set_visible(False)
                axp.spines['left'].set_visible(False)
                axp.spines['bottom'].set_visible(False)
                axp.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)
                cpwm = clusterpwms[l].T
                #cpwm = np.log2((cpwm+1e-10)/0.25)
                #cpwm[cpwm < 0] = 0
                cpwmnames = motifnames[t][l]
                cpwm = pd.DataFrame({'A':cpwm[0],'C':cpwm[1], 'G':cpwm[2], 'T':cpwm[3]})
                lm.Logo(cpwm, ax = axp)
                axp.set_xlabel(cpwmnames,rotation = 0)
        
        fig.savefig(outname +'_motif_cluster_counts.jpg', dpi = 150, bbox_inches = 'tight')
        print(outname +'_motif_cluster_counts.jpg')
    
    
    if '--plot_scatter' in sys.argv and len(target_tracks) == 2:
        # can extended to all versus all
        
        allpwmcounts = np.sum(allpwmcounts[:,1],axis = 1)
        #print(np.shape(allpwmcounts))
        uniqueones, uniqueind, uniquenum = np.unique(allpwmcounts, return_counts = True,return_index = True, axis = 1)
        #print(uniqueones, uniquenum)
        
        fig = plt.figure(figsize = (4,4),dpi = 200)
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('# sequences with motif')
        ax.set_xlabel(ctype[0]+'('+str(np.mean(target_values[0]))+')')
        ax.set_ylabel(ctype[1]+'('+str(np.mean(target_values[1]))+')')
        ax.scatter(uniqueones[0], uniqueones[1], s = 40*uniquenum, color = 'slategrey')
        ax.plot([np.amin(allpwmcounts),np.amax(allpwmcounts)],[np.amin(allpwmcounts),np.amax(allpwmcounts)], color = 'k')
        for u, us in enumerate(uniqueones.T):
            if uniquenum[u] > 1:
                ax.text(us[0], us[1], str(uniquenum[u]), ha = 'center', va = 'center', fontsize = 6, color = 'white')
        fig.savefig(outname +'_motif_cluster_countscatter.jpg', dpi = 400, bbox_inches = 'tight')
        print(outname +'_motif_cluster_countscatter.jpg')
    
    if '--plot_interactions' in sys.argv:
        # for each condition test with fisher's exact motif interaction score in generated sequences, maybe start with most enriched motifs and use threshold to avoid n^2 time.
        minoccur = 2
        motif_interactions = [[] for t in range(len(target_tracks))]
        min_pval = 0.01
        for t in range(len(target_tracks)):
            pwmcount = pwmcounts[t][1][0]
            sortpwms = np.argsort(-pwmcount)
            pwin = pwmsin[t]
            for g, s in enumerate(sortpwms):
                for q in sortpwms[g:]:
                    if pwmcount[s] >= minoccur and pwmcount[q] >= minoccur:
                        pws, pwq = pwin[s], pwin[q]
                        clmasks = clusters == pws
                        clmaskq = clusters == pwq
                        tmask = (pwmstats[:,0] ==t) * (pwmstats[:,1] ==1) * (pwmstats[:,2] ==1)
                        clmasks, clmaskq = clmasks*tmask, clmaskq*tmask
                        seqs, seqsn = np.unique(pwmstats[clmasks, -1], return_counts = True) 
                        seqq, seqqn= np.unique(pwmstats[clmaskq, -1], return_counts = True)
                        if s == q:
                            combseqs = seqs[seqsn > 1]
                        else:
                            combseqs = np.intersect1d(seqs, seqq)
                        ss, sq, cs = len(seqs), len(seqq), len(combseqs)
                        table = [[ss-cs,cs],[Nseqs-ss-sq+cs,sq-cs]]
                        odd_, pval_interactg = fisher_exact(table, alternative = 'greater')
                        odd_, pval_interactl = fisher_exact(table, alternative = 'less')
                        #print(ss, sq, cs, pval_interactg, pval_interactl)
                        if min(pval_interactg, pval_interactl) < min_pval:
                            motif_interactions[t].append([pws, pwq, -np.log10(pval_interactl)+np.log10(pval_interactg), cs])
                            print('motifinterractions', motif_interactions[t][-1])
        
        if len(np.concatenate(motif_interactions, axis = 0)) > 0:
            # Generate figure with log10(p-value as bar plot and motifs on left side
            maxmotint = max(10,np.amax([len(mit) for mit in motif_interactions]))
            cols, rows = 4*len(motif_interactions), 1*maxmotint + 1
            fig = plt.figure(figsize = (cols, rows), dpi = 150)
            for m, mit in enumerate(motif_interactions):
                mit = np.array(mit)
                if len(mit) > 0:
                    motif_interactions[m] = mit[np.argsort(mit[:,2])]
            for t in range(len(target_tracks)):
                if len(motif_interactions[t]) > 0:
                    
                    motblocky = 0.8/maxmotint
                    motblockx = 2*rows/cols*0.8/maxmotint
                    
                    ax = fig.add_subplot(1, len(target_tracks)*3, t*3+3)
                    ax.set_position([0.1+t*0.8/len(target_tracks)+2*0.65*motblockx,0.9-len(motif_interactions[t])*motblocky,0.8/len(target_tracks)-2*0.65*motblockx,len(motif_interactions[t])*motblocky])
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.tick_params(left = False, labelleft = False)
                    ax.barh(-np.arange(len(motif_interactions[t])), motif_interactions[t][:,2], color = 'purple', alpha = 0.8)
                    ax.set_title(ctype[t]+'('+str(np.mean(target_values[t])) + ')')
                    ax.set_xlabel('-log10(p-value)')
                    ax.set_ylim([-len(motif_interactions[t])+0.5,0.5])
                    ax.set_xticks([np.log10(0.001),np.log10(0.01), np.log10(0.05), -np.log10(0.05),-np.log10(0.01), -np.log10(0.001)])
                    ax.grid(axis = 'x')
                    ax.set_xticklabels(['***', '**', '*', '*', '**', '***'])
                    for i,j in enumerate(-np.arange(len(motif_interactions[t]))):
                        ax.text(motif_interactions[t][i,2]+np.sign(motif_interactions[t][i,2])*0.2,j, str(int(motif_interactions[t][i,3])))
                    for l, moti in enumerate(motif_interactions[t]):
                        axp1 = fig.add_subplot(maxmotint,len(motif_interactions)*3, 3*len(motif_interactions)*l+1+t*3)
                        axp2 = fig.add_subplot(maxmotint,len(motif_interactions)*3, 3*len(motif_interactions)*l+2+t*3)
                                    
                        
                        axp1.set_position([0.1+t*0.8/len(target_tracks)+0.03*motblockx, 0.9-(l+1)*0.8/maxmotint+0.2*0.8/maxmotint, 0.6*motblockx, 0.6*motblocky])
                        axp2.set_position([0.1+t*0.8/len(target_tracks)+1.*0.68*motblockx, 0.9-(l+1)*0.8/maxmotint+0.2*0.8/maxmotint, 0.6*motblockx, 0.6*motblocky])
                        
                        axp1.spines['right'].set_visible(False)
                        axp1.spines['top'].set_visible(False)
                        axp1.spines['left'].set_visible(False)
                        axp1.spines['bottom'].set_visible(False)
                        axp1.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)
                        
                        axp2.spines['right'].set_visible(False)
                        axp2.spines['top'].set_visible(False)
                        axp2.spines['left'].set_visible(False)
                        axp2.spines['bottom'].set_visible(False)
                        axp2.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)
                        
                        lp1 = list(unclusters).index(moti[0])
                        lp2 = list(unclusters).index(moti[1])
                        
                        
                        cpwm1 = clusterpwms[lp1].T
                        #cpwm1 = np.log2((cpwm1+1e-10)/0.25)
                        #cpwm1[cpwm1 < 0] = 0
                        cpwmnames1 = motifnames[t][lp1]
                        cpwm1 = pd.DataFrame({'A':cpwm1[0],'C':cpwm1[1], 'G':cpwm1[2], 'T':cpwm1[3]})
                        lm.Logo(cpwm1, ax = axp1)
                        axp1.set_xlabel(cpwmnames1, fontsize = 10)
                        
                        cpwm2 = clusterpwms[lp2].T
                        #cpwm2 = np.log2((cpwm2+1e-10)/0.25)
                        #cpwm2[cpwm2 < 0] = 0
                        cpwmnames2 = motifnames[t][lp2]
                        cpwm2 = pd.DataFrame({'A':cpwm2[0],'C':cpwm2[1], 'G':cpwm2[2], 'T':cpwm2[3]})
                        lm.Logo(cpwm2, ax = axp2)
                        axp2.set_xlabel(cpwmnames2, fontsize = 10)
                        
            fig.savefig(outname +'_motif_cluster_combinations.jpg', dpi = 200, bbox_inches = 'tight')        
            print(outname +'_motif_cluster_combinations.jpg')

