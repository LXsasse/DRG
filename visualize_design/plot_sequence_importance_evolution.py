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
from compare_expression_distribution import read_separated
import time

# Read text files with PWMs
def read_pwm(pwmlist, nameline = 'Motif'):
    names = []
    pwms = []
    features = {}
    collected_features = {}
    pwm = []
    obj = open(pwmlist, 'r').readlines()
    for l, line in enumerate(obj):
        line = line.strip('\n').split('\t')
        #print(line)
        if ((len(line) == 0) or (line[0] == '')) and len(pwm) > 0:
            pwm = np.array(pwm, dtype = float)
            pwms.append(np.array(pwm))
            pwm = []
            names.append(name)
            for fe in features:
                if fe in collected_features.keys():
                    collected_features[fe].append(features[fe])
                else:
                    collected_features[fe] = [features[fe]]
            features = {}
        elif len(line) > 0:
            if line[0] == nameline:
                name = line[1]
                pwm = []
            if line[0] == 'Pos':
                nts = line[1:]
            elif isinstance(numbertype(line[0]), int):
                pwm.append(line[1:])
            elif line[0] != '':
                features[line[0]] = line[1]
    return pwms, names, collected_features

def bestpwmmatch(saliency, st, en, pfms, infocont = True):
    matchscore = []
    matchpos = []
    lcontrol = []
    saliency = saliency.to_numpy()
    osaliency = np.copy(saliency)
    saliency[np.sign(np.sum(saliency[st:en]))*saliency < 0] = 0
    saliency = np.absolute(saliency)
    for p, pfm in enumerate(pfms):
        if infocont:
            pfm = np.log2((pfm+1e-16)/0.25)
        ### Somehow devide by sum over max of all positions, to control for limited usage tf pwm
        pfmnorm = np.sum(np.amax(pfm, axis = 1))
        #print(pfmnorm)
            
        #pfm[pfm<0] = 0
        #print('Before', len(pfm), en-st)
        extend = int(min(en-st,len(pfm))/4) + int(min(en-st,len(pfm))%4!=0) + max(0,len(pfm) - en +st)
        #print(extend, np.shape(saliency))
        spwmin = np.amin(np.sum(saliency,axis = 1)[st:en])
        #print(spwmin)
        spwm = np.ones((en-st+2*extend,4))*(spwmin/8.)
        #print(np.shape(spwm))
        #print(max(0,extend-st), max(0,en+extend-len(saliency)))
        spwm[max(0,extend-st):len(spwm)-max(0,en+extend-len(saliency))] = saliency[max(0,-extend+st):min(len(saliency),en+extend)]
        #print('Compare', len(spwm),len(pfm))
        mscore = -10000
        mp = 0
        lc = 0
        #print(len(spwm)-len(pfm))
        for l in range(len(spwm)-len(pfm)):
            #print(spwm[l:l+len(pfm)], np.sum(pfm*spwm[l:l+len(pfm)]),  np.sum(spwm[l:l+len(pfm)]))
            #print(np.sum(spwm[l:l+len(pfm)],axis = 1) > 0)
            #print(np.amax(pfm[np.sum(spwm[l:l+len(pfm)],axis = 1) > 0],axis = 1))
            
            pnorm = np.sum(np.amax(pfm[np.sum(spwm[l:l+len(pfm)],axis = 1) > 0],axis = 1))
            #print(pnorm)
            pnorm/=pfmnorm
            #print(pnorm)
            #sys.exit()
            #print(pfm*spwm[l:l+len(pfm)], spwm[l:l+len(pfm)])
            ms = np.sum(pfm*spwm[l:l+len(pfm)])/np.sum(spwm[l:l+len(pfm)])
            ms *= pnorm
            #print(l, ms)
            if ms > mscore:
                mscore = ms
                mp = max(0,-extend+st) + l
                lc = l
        #print('MSCORE', mscore)
        matchscore.append(mscore)
        matchpos.append(mp)
        lcontrol.append(lc)
    
    argmax = np.argsort(matchscore)[::-1]
    #print(np.array(matchscore)[argmax][:10])
    '''
    for p in argmax[:3]:
        print(matchscore[p])
        pfm = pfms[argmax[0]]
        pfm = np.log2((pfm+1e-16)/0.25)
        extend = int(min(en-st,len(pfm))/4) + int(min(en-st,len(pfm))%4!=0) + max(0,len(pfm) - en +st)
        spwmin = np.amin(np.sum(saliency,axis = 1)[st:en])
        spwm = np.ones((en-st+2*extend,4))*(spwmin/8.)
        spwm[max(0,extend-st):len(spwm)-max(0,en+extend-len(saliency))] = saliency[max(0,-extend+st):min(len(saliency),en+extend)]
        spwm2 = np.copy(spwm)
        spwm2[max(0,extend-st):len(spwm)-max(0,en+extend-len(saliency))] = osaliency[max(0,-extend+st):min(len(saliency),en+extend)] 
        l = lcontrol[p]
        nts = np.array(list('ACGT'))
        
        print(pfm)
        print(spwm[l:l+len(pfm)])
        print(spwm2[l:l+len(pfm)])
        print(np.sign(np.sum(saliency[st:en])))
        print(rbpnames[p], ''.join(nts[np.argmax(pfm,axis = 1)]), ''.join(nts[np.argmax(saliency[st:en],axis = 1)]))
    '''
    maxscore, maxoff = np.array(matchscore)[argmax], np.array(matchpos)[argmax]
    #print(maxscore, maxoff)
    return maxscore, maxoff, argmax

            
def find_pwms(saliency, pfms, z_cut = 1.3, detect_cut = 1., min_mot = 5, infocont = True, equal_score = True, mean_sal = 0):
    salpwms = None
    # find positions in saliency that represent motifs
    if mean_sal is None:
        zscore_sal = (np.sum(saliency, axis =1) - np.mean(np.sum(saliency,axis = 1)))/np.std(np.sum(saliency,axis = 1))
    else:
        zscore_sal = (np.sum(saliency, axis =1) - mean_sal)/np.sqrt(np.mean((np.sum(saliency,axis = 1)-mean_sal)**2))
    potmotifs = np.absolute(zscore_sal) >= z_cut
    zsign = np.sign(zscore_sal) * potmotifs
    lmot, lgap,ngap, ldif, lstr, ps, sig = 0, 0, 0, 0, 0, None, None
    # detemine motifs in saliency map as positions with enriched saliency scores
    salpstart, salpend = [], []
    #print(np.where(potmotifs))
    # acount for different signs
    # find continous stretches with same sign # allow one gap per 7 basepairs, minimum continous stretch must be 3 within and 1 one the flanks
    p = 0
    while True:
    #for p in range(len(potmotifs)):
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
        #print(p, lmot, lgap, ngap, lstr, sig)
        p += 1
        if p == len(potmotifs):
            break
    
    #print(salpstart, salpend)
    #sys.exit()
    salpwms = saliency.copy()
    #print(salpwms[salpstart[0]-2:salpend[0]+2+1])
    salpwms[:] = 0
    salpwmsreg = np.ones(len(salpwms))
    salpos, salnames = [],[]
    if len(salpstart) > 0:
        # find best matching pfms
        for s in range(len(salpstart)):
            if equal_score:
                salmap = np.sign(saliency)
                #salmap[~potmotifs] = 0
            else:
                salmap = saliency
            mscore, mpos, argpwm = bestpwmmatch(salmap, salpstart[s], salpend[s]+1, pfms, infocont = infocont)
            for m, ms in enumerate(mscore):
                if ms > detect_cut:
                    pfmchose = pfms[argpwm[m]]
                    if np.sum(np.sum(salpwms[mpos[m]:mpos[m]+len(pfmchose)],axis =1) > 0)/len(pfmchose) < 0.25:
                        if infocont:
                            pfmchose = np.log2((pfmchose+1e-10)/0.25)
                            pfmchose[pfmchose<0] = 0
                        salpwms[mpos[m]:mpos[m]+len(pfmchose)] += pfmchose
                        salpwmsreg[mpos[m]:mpos[m]+len(pfmchose)] += 1
                        salpos.append(mpos[m]+len(pfmchose)/2)
                        salnames.append(argpwm[m])
    salpwmsreg -= 1
    salpwmsreg[salpwmsreg == 0] = 1
    salpwms /= salpwmsreg[:,None]
    return salpwms, salnames, salpos, salpstart, salpend




def saliency_map(cseq, target, loss_function, model, scoring):
    loss = loss_dict[loss_function]
    cseq, target = torch.Tensor(cseq), torch.Tensor(target).unsqueeze(0)
    
    if scoring == 'gradient':
        loss.reduction = 'mean'
        cseq.requires_grad = True
        pred = model.forward(cseq)
        if cseq.grad is not None:
            cseq.grad.data.zero_()
    
        diffloss = loss(pred, target)
        diffloss = torch.mean(diffloss)
        gradin = diffloss.backward(retain_graph = True)
        grad = -cseq.grad.numpy()
        saliency = grad[0]*cseq[0].detach().numpy()
    
    elif scoring == 'forward':
        
        #t0 = time.time()
        # extend zero axis for cseq,  then set all positions to 0 individually and then insert the one all at once
        setto = np.array(np.where(cseq[0] == 0))
        icur = np.array(np.where(cseq[0] == 1))
        with torch.no_grad():
            target = target.expand(len(setto[0])+1,-1)
            tseq = torch.clone(cseq.detach())
            tseq = tseq.repeat(len(setto[0])+1,1,1)
            iscur = []
            for i in range(len(setto[0])):
                iscur.append([icur[0][icur[1] == setto[1][i]][0] , setto[1][i]])
                tseq[i, :, setto[1][i]] = 0
                tseq[i, setto[0][i], setto[1][i]] = 1
            iscur = np.array(iscur).T
            #print(np.where(np.where(tseq[0].T == 1)[1] != np.where(tseq[-1].T == 1)[1]), setto[0][0], setto[1][0])
            #print(np.where(np.where(tseq[300].T == 1)[1] != np.where(tseq[-1].T == 1)[1]), setto[0][300], setto[1][300])
            tpred = model.forward(tseq)
            ismloss = loss(tpred,target)
            ismloss = torch.mean(ismloss, axis = 1)
            saliency = np.zeros(cseq[0].size())
            diffloss = ismloss[-1]
            for i in range(len(setto[0])):
                saliency[iscur[0][i], iscur[1][i]] += ismloss[i]-diffloss
            pred = tpred[[-1]]
                #print(setto[0][i], setto[1][i], ismloss[i], diffloss)
            #print('Saliency', saliency)
            ##t1 = time.time()
            #print(t1-t0)
        '''
        #t0 = time.time()
        loss.reduction = 'mean'
        pred = model.forward(cseq)
        lenseq = cseq.size(dim=-1)
        with torch.no_grad():
            diffloss = loss(pred, target)
            diffloss = torch.mean(diffloss)
            saliency = np.zeros(cseq.size())
            for l in range(lenseq):
                iscur = np.where(cseq[...,l] == 1)[1][0]
                setto1 = np.where(cseq[...,l] == 0)[1]
                for i in setto1:
                    tseq = torch.clone(cseq).detach()
                    tseq[0,i,l] = 1
                    tseq[0,iscur, l] = 0
                    #print(np.where(tseq[0].T == 1)[1] , np.where(cseq[0].T == 1)[1])
                    #print(np.where(np.where(tseq[0].T == 1)[1] != np.where(cseq[0].T == 1)[1]), i,l)
                    tpred = model.forward(tseq)
                    ismloss = torch.mean(loss(tpred, target))
                    saliency[0,iscur,l] += ismloss.item()-diffloss.item()
                    #print(i,l,ismloss, diffloss)
        saliency = saliency[0]
        #print('Saliency', saliency)
        #t1 = time.time()
        #print(t1-t0)
        '''
    return saliency, diffloss.item(), pred.detach().numpy()[0]

def transvalue(strval):
    if '--' in strval:
        strval = strval.split("-")
        return np.array(['-'.join(strval[:-2]), '-'.join(strval[-2:])], dtype =float)
    else:
        return np.array(strval.rsplit('-',1), dtype = float)

if __name__ == '__main__':
    seed_names, seed_seqs = readinfasta(sys.argv[1])

    evofile = sys.argv[2]
    outname = os.path.splitext(evofile)[0]
    obj = open(evofile, 'r').readlines()
    opt = []
    opt_pos = []
    opt_names = []
    opt_vals = []
    for l, line in enumerate(obj):
        line = line.strip().split(';')
        name = line[0].split(',')[0]
        opt_names.append(name)
        seedseq = seed_seqs[list(seed_names).index(name)]
        op = [seedseq]
        opp = [None]
        opv = [transvalue(line[0].split(',')[1])]
        line = line[1:]
        for ent in line:
            ent = ent.split(',')
            pos = int(ent[0])
            opp.append(pos)
            orig, new = ent[1].split('/')
            seedseq = seedseq[:pos]+new+seedseq[pos+1:]
            op.append(seedseq)
            opv.append(transvalue(ent[2]))
        opt.append(op)
        opt_pos.append(opp)
        opt_vals.append(opv)
    
    stepsize = int(sys.argv[3])

    predictor = sys.argv[4]

    target_tracks, tset_tracks = read_separated(sys.argv[5])
    
    ctype = sys.argv[6]
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
    
    target_values = sys.argv[7]
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

    scoring = sys.argv[8]

    loss_function = sys.argv[9]

    outname += '_'+scoring+'_'+loss_function

    pwmline = False
    if '--search_pwms' in sys.argv:
        pwmfile = sys.argv[sys.argv.index('--search_pwms')+1]
        pfms, rbpnames, rbpfeat = read_pwm(pwmfile, nameline = 'TF Name')
        rbpnames = np.array([rname.split(';')[0] if ';' in rname else rname for rname in rbpnames])
        pfms = np.array(pfms)
        
        pwmline = True
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
        print('Considerable PFMs', np.sum(considerpfm, axis = 0))
        pwmcount = [[[],[]] for i in range(len(target_tracks))]
        
        
    for q in range(0, len(opt)):
        print('Seed',q)
        seqset = opt[q]
        columns = len(target_tracks)
        rows = int(len(seqset)/stepsize) + 1 + int(len(seqset)%stepsize != 0)
        if pwmline:
            rows*=2
        fig = plt.figure(figsize = (12*columns, 1.*rows))
        seqset, nts = quick_onehot(seqset)
        seqset = np.transpose(seqset, axes=(0,2,1))
        for t, tt in enumerate(target_tracks):
            model = load_cnn_model(predictor, verbose = False)
            model.classifier.Linear.weight = nn.Parameter(model.classifier.Linear.weight[[tt]])
            model.classifier.Linear.bias = nn.Parameter(model.classifier.Linear.bias[[tt]])
            steps = np.arange(len(seqset), dtype = int)
            steps = steps[(steps%stepsize == 0) | (steps == np.amax(steps))]
            pwms = []
            preds = []
            for i, s in enumerate(steps):
                #print(columns*i+1+t, i, t)
                ohseq = seqset[[s]]
                saliency, loss, pred = saliency_map(ohseq, target_values[t], loss_function, model, scoring)
                print(np.mean(pred), opt_vals[q][s][-1-t])
                pwm = pd.DataFrame({'A':saliency[0],'C':saliency[1], 'G':saliency[2], 'T':saliency[3]})
                pwms.append(pwm)
                preds.append(np.mean(pred))
            ylim = [np.amin(pwms), np.amax(pwms)]
            #print(ylim)
            #print(steps, columns, rows, len(seqset))
            j = 1
            for i, s in enumerate(steps):
                #print(rows, columns, columns*i+1+t)
                pwm = pwms[i]
                if pwmline:
                    j = 2
                    # search for motifs of length 6 or more in saliency saliency_map
                    salpwms, salnames, salpos, salpstart, salpend = find_pwms(pwm, pfms[considerpfm[:,t]], z_cut = 1.3, detect_cut = 1., infocont = True, equal_score = False)
                    if len(salnames) > 0:
                        if i == 0:
                            pwmcount[t][0].append(np.where(considerpfm[:,t])[0][salnames])
                        elif i == len(steps)-1:
                            pwmcount[t][1].append(np.where(considerpfm[:,t])[0][salnames])
                        #print('axp',t, i, j, columns*(i*j)+1+t)
                        axp = fig.add_subplot(rows, columns, columns*(i*j)+1+t)
                        posaxp = axp.get_position()
                        axp.set_position([posaxp.x0, posaxp.y0 +posaxp.height *0.3, posaxp.width, posaxp.height *0.6])
                        lm.Logo(salpwms, ax = axp)
                        axp.spines['top'].set_visible(False)
                        axp.spines['right'].set_visible(False)
                        axp.spines['left'].set_visible(False)
                        #axp.spines['bottom'].set_visible(False)
                        axp.tick_params(bottom = True, labelbottom = True, left = False, labelleft = False)
                        axp.set_xticks(salpos)
                        axp.set_xticklabels(rbpnames[np.where(considerpfm[:,t])[0][salnames]])
                    
                        
                #print('ax',t, i, j, columns*(i*j+1)+1+t)
                ax = fig.add_subplot(rows, columns, columns*(i*j+1)+1+t)
                if opt_pos[q][s] is not None:
                    #if s == 0:
                        #ssb = [opt_pos[q][s]]
                    #else:
                    ssb = np.unique(opt_pos[q][1:s+1])
                    ax.bar(ssb, np.ones(len(ssb))*(ylim[1]-ylim[0]), width = 1., bottom = ylim[0], alpha = 0.2, color = 'purple')
                
                if pwmline:
                    for sp in range(len(salpstart)):
                        sumsal = np.sum(pwm[salpstart[sp]:salpend[sp]+1],axis =1).to_numpy()
                        ax.bar(salpstart[sp]-0.5, sumsal[np.argmax(np.absolute(sumsal))], width = salpend[sp] +1 - salpstart[sp], color = 'none', align ='edge', edgecolor = 'k')
                
                lm.Logo(pwm, ax = ax)
                ax.set_ylabel(str(s)+': '+str(np.around(preds[i],1)))
                ax.set_ylim(ylim)
                #print(ylim)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if i != len(steps)-1:
                    ax.tick_params(bottom = False, labelbottom = False)
                if i == 0:
                    if pwmline and len(salnames) > 0:
                        axp.set_title(ctype[t])
                    else:
                        ax.set_title(ctype[t])
        
        fig.savefig(outname +opt_names[q]+'_saliencymap.jpg', dpi = 200, bbox_inches = 'tight')
        print(outname +opt_names[q]+'_saliencymap.jpg')
        plt.close()
        if q == 99 or q == len(opt)-1:
            for t in range(len(target_tracks)):
                print(ctype[t], target_values[t])
                print('Beginning')
                rib, nr = np.unique(rbpnames[np.concatenate(pwmcount[t][0])], return_counts = True)
                for r, ri in enumerate(rib):
                    print(ri, nr[r])
                print('Optimized')
                rib, nr = np.unique(rbpnames[np.concatenate(pwmcount[t][1])], return_counts = True)
                for r, ri in enumerate(rib):
                    print(ri, nr[r])
            sys.exit()
            
        







