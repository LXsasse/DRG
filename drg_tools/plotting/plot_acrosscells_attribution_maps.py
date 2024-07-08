import numpy as np
import sys, os
import matplotlib.pyplot as plt
import logomaker as lm
import pandas as pd
from scipy.stats import pearsonr


def add_frames(att, locations, colors, ax):
    att = np.array(att)
    cmap = ['purple', 'limegreen']
    for l, loc in enumerate(locations):
        mina, maxa = np.amin(np.sum(np.ma.masked_greater(att[loc[0]:loc[1]+1],0),axis = 1)), np.amax(np.sum(np.ma.masked_less(att[loc[0]:loc[1]+1],0),axis = 1))
        x = [loc[0]-0.5, loc[1]+0.5]
        ax.plot(x, [mina, mina], c = cmap[colors[l]])
        ax.plot(x, [maxa, maxa], c = cmap[colors[l]])
        ax.plot([x[0], x[0]] , [mina, maxa], c = cmap[colors[l]])
        ax.plot([x[1], x[1]] , [mina, maxa], c = cmap[colors[l]])


def logoax(fig, att, ylabel = None, ylim = None, sb = 111, pos = None, labelbottom = True, bottom = True, xticks = None, xticklabels = None):
    ax0 =  fig.add_subplot(sb[0], sb[1], sb[2])
    if pos is not None:
        ax0.set_position(pos)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.tick_params(bottom = bottom, labelbottom = labelbottom)
    att = pd.DataFrame({'A':att[:,0],'C':att[:,1], 'G':att[:,2], 'T':att[:,3]})
    lm.Logo(att, ax = ax0)
    if ylabel is not None:
        ax0.set_ylabel(ylabel)
    if ylim is not None:
        ax0.set_ylim(ylim)
    if xticks is not None:
        ax0.set_xticks(xticks)
    if xticklabels is not None:
        ax0.set_xticklabels(xticklabels)
    return ax0
    
def heatax(ism, fig, pos = None, sb = 111, cmap = 'coolwarm', ylabel = None, labelbottom = True, bottom = True, vlim = None):
    if vlim is None:
        vlim = np.amax(np.absolute(ism))
    ax1 =fig.add_subplot(sb)
    if pos is not None:
        ax1.set_position(pos)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.imshow(ism.T, aspect = 'auto', cmap = cmap, vmin = -vlim, vmax = vlim)
    if ylabel is not None:
        ax1.set_ylabel(ylabel)
    ax1.tick_params(bottom = bottom, labelbottom = labelbottom)
    ax1.set_yticks(np.arange(4))
    ax1.set_yticklabels(['A','C','G','T'])
    return ax1

def activity_plot(values, ylim, xticklabels, ax):
    ax.bar(np.arange(len(values)), values)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(ylim)
    if xticklabels is None:
        ax.tick_params(bottom = False, labelbottom = False)
    else:
        ax.set_xticklabels(xticklabels, rotation = 60)
    return ax

def generate_xticks(start, end, n):
    possible = np.concatenate([np.array([1,2,5,10])*10**i for i in range(-16,16)])
    steps=(end-start)/n
    steps = possible[np.argmin(np.absolute(possible - steps))]
    ticklabels = np.arange(start, end)
    ticks = np.where(ticklabels%steps == 0)[0]
    ticklabels = ticklabels[ticks]
    return ticks, ticklabels
    
    

def plot_attribution(seq, att, motifs = None, seq_based = 1, exp = None, vlim = None, unit = 0.15, ratio = 10, ylabel = None, xtick_range = None, barplot = None):
    #print(att[0,:10,:,0], att[0,:10,:,0])
    ism = np.copy(att)
    if seq_based:
        att = seq * att
        ylabel = 'Attribution\nat ref'
    
    if ylabel is None:
        ylabel = 'Attribution'
    
    if exp is None:
        exp = np.arange(len(att), dtype = int).astype(str)
        
    if vlim is None:
        mina = min(0,np.amin(np.sum(np.ma.masked_greater(att,0), axis = -1)))
        maxa = np.amax(np.sum(np.ma.masked_less(att,0), axis = -1))
        attlim = [mina, maxa]
    else:
        attlim = vlim
    
    if xtick_range is not None:
        xticks, xticklabels = generate_xticks(xtick_range[0], xtick_range[1], 7)
    else:
        xticks, xticklabels = None, None
    
    fig = plt.figure(figsize = (unit*len(seq), len(att) * ratio*unit), dpi = 50)
    
    axs = []
    for a, at in enumerate(att):
        axs.append(logoax(fig, at, ylabel = exp[a], ylim = attlim, sb = [len(att), 1, 1+a], pos = [0.1,0.1+(len(att)-1-a)/len(att)*0.8,0.8,0.8*(1/len(att))*0.8], labelbottom = a == len(att)-1, bottom = a == len(att)-1, xticks = xticks, xticklabels = xticklabels))
    
    # This is for a barplot on the side of the sequence logo, that shows predicted and/or measured actibity
    if barplot is not None:
        ylim = [0, np.amax(barplot)]
        for b, bp in enumerate(barplot):
            ax = fig.add_subplot(len(barplot), len(barplot), len(barplot) + b)
            ax.set_position([0.9 + 2.5*0.8*(1/len(seq)), 0.1+(len(att)-1-b)/len(att)*0.8, 6*0.8*(1/len(seq)), 0.8*(1/len(att))*0.8])
            axs.append(activity_plot(bp, ylim, None, ax))
    
    
    if motifs is not None:
        mask = motifs[:,-2] == 0
        colors = motifs[mask,-1]
        #print(motifs[mask,1])
        locations = [ti1[l] for l in motifs[mask,1]]
        #print(locations)
        add_frames(att, locations, colors, ax0)

    return fig


def isint(x):
    try:
        int(x) 
        return True
    except:
        return False


def get_indx(given, target, islist = False):
    if islist:
        if given == 'all':
            indx = np.arange(len(target), dtype = int)
        else:
            if isinstance(given, str):
                indx = given.split(',')
            for e, el in enumerate(indx):
                if isint(el):
                    indx[e] = int(el)
                else:
                    indx[e] = list(target).index(el)
    else:
        if isint(given):
            indx = int(given)
        else:
            indx = list(target).index(given) 
            
    return indx
    

def read_txt(txtfile, delimiter = None):
    lines = open(txtfile, 'r').readlines()
    exp, names, data = None, [],[]
    for l, line in enumerate(lines):
        if l ==0 and '#' == line[0]:
            exp = line.strip('#').strip().split(delimiter)
        else:
            line = line.strip().split(delimiter)
            names.append(line[0])
            data.append(np.array(line[1:], dtype = float))
    if exp is not None:
        exp = np.array(exp)
    data, names = np.array(data),np.array(names)
    
    return names, data, exp
            
            
            
            
            
            

if __name__ == '__main__':
    ism = np.load(sys.argv[1], allow_pickle = True)
    ref = np.load(sys.argv[2], allow_pickle = True)
    
    selectin = sys.argv[3]
    electin = sys.argv[4]
    
    
    names, values, exp = ism['names'], ism['values'], ism['experiments']
    if len(np.shape(values)) >4:
        values = values[int(sys.argv[5])]
    
    seqfeatures, genenames = ref['seqfeatures'], ref['genenames']
    if len(np.shape(seqfeatures)) == 1:
        seqfeatures, featurenames = seqfeatures
    
    nsort = np.argsort(names)[np.isin(np.sort(names), genenames)]
    gsort = np.argsort(genenames)[np.isin(np.sort(genenames), names)]
    values = values[nsort]
    seqfeatures = seqfeatures[gsort]
    names, genenames = names[nsort], genenames[gsort]
    values = np.transpose(values, axes = (0,1,3,2))
    outname = os.path.splitext(sys.argv[1])[0] +'_'+selectin+'_'+electin
    
    if '--outname' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]
    
    select = get_indx(selectin, names)
    elect = get_indx(electin, exp, islist = True)
    electin = ','.join(exp[elect])
    
    seq = seqfeatures[select]
    
    att = values[select]
    att = att[elect]
    
    exp = exp[elect]
    print(names[select], genenames[select], exp)
    print(np.shape(att))
    
    if np.shape(att)[-1] != np.shape(seq)[-1]:
        nshape = list(np.shape(seq))
        nshape = [len(att)] + nshape
        natt = np.zeros(nshape)
        for a, at in enumerate(att):
            natt[a,at[:,-1].astype(int)] = at[:,:np.shape(seq)[-1]]
        att = natt
    
    maxatt = np.amax(att, axis = (-2,-1))
    
    newrange = None
    if '--remove_zero_attributions' in sys.argv:
        mask = np.where(np.sum(np.absolute(att),axis = (0,-1)))[0]
        newrange = [mask[0], mask[-1]]
        print('New range', newrange)
        att = att[:,newrange[0]:newrange[1]+1]
        seq = seq[newrange[0]:newrange[1]+1]
    
    if '--remove_low_attributions' in sys.argv:
        lowness = float(sys.argv[sys.argv.index('--remove_low_attributions')+1])
        mask = np.amax(np.absolute(att),axis = -1)
        mask = np.amax(mask, axis = 0)
        mask = mask > lowness * np.amax(mask)
        mask = np.where(mask)[0]
        newrange = [mask[0], mask[-1]]
        print('New range', newrange)
        att = att[:,newrange[0]:newrange[1]+1]
        seq = seq[newrange[0]:newrange[1]+1]
    
    if '--centerofmass_attributions' in sys.argv:
        window = int(sys.argv[sys.argv.index('--centerofmass_attributions')+1])
        com = np.argmax(np.convolve( np.sum(np.absolute(att), axis = (0,-1)), np.ones(7,dtype=int), 'full'))
        newrange = [max(0,com-window), min(com+window,len(seq))]
        print('New range', newrange)
        att = att[:,newrange[0]:newrange[1]+1]
        seq = seq[newrange[0]:newrange[1]+1]
        
    
    mlocs = None
    if '--motif_location' in sys.argv:
        mlocfile = np.genfromtxt(sys.argv[sys.argv.index('--motif_location')+1], dtype = str)[:,0]
        mlocs = np.array([m.rsplit('_',3) for m in mlocfile])
        keep = [i for i, name in enumerate(mlocs[:,0]) if name in sys.argv[1]]
        mlocs = mlocs[keep].astype(object)
        for m, ml in enumerate(mlocs):
            mlocs[m][1] = np.array(ml[1].split('-'), dtype = int)
        mlocs[:,[2,3]] = mlocs[:,[2,3]].astype(int)

    
    if '--centerattributions' in sys.argv:
        att -= (np.sum(att, axis = -1)/4)[...,None]
    elif '--decenterattributions' in sys.argv:
        att -= seq * att
    elif '--meaneffectattributions' in sys.argv:
        att -= (np.sum((seq == 0)*att, axis = -1)/3)[...,None]
    
    if '--difference' in sys.argv:
        diff = int(sys.argv[sys.argv.index('--difference')+1])
        attd = att[diff]
        mask = np.arange(np.shape(att)[0]) != diff
        att = att[mask]
        att -= attd[None]
        expd = exp[diff]
        exp = exp[mask]
        exp = np.array([e +'\n-'+expd for e in exp])
    
    seq_based = True
    if '--showall_attributions' in sys.argv:
        seq_based = False

    barplot = None
    if '--add_predictions' in sys.argv:
        predictions = np.load(sys.argv[sys.argv.index('--add_predictions')+1])
        pnames, pvalues, pcolumns = predictions['names'], predictions['values'], predictions['columns']
        pselect = get_indx(selectin, pnames)
        pelect = get_indx(electin, pcolumns, islist = True)
        pnames, pvalues, pcolumns = pnames[pselect], pvalues[pselect][pelect], pcolumns[pelect]
        barplot = [[pv] for pv in pvalues]
        print(pearsonr(np.array(barplot)[:,0], maxatt))
        
    if '--add_measured' in sys.argv:
        measured = sys.argv[sys.argv.index('--add_measured')+1]
        if os.path.splitext(measured)[1] == '.npz':
            measured = np.load(sys.argv[sys.argv.index('--add_measured')+1])
            mnames, mvalues, mcolumns = measured['names'], measured['counts'], measured['celltypes']
        else:
            mnames, mvalues, mcolumns = read_txt(measured)
        if '--add_measured_apendix' in sys.argv:
            appendix = sys.argv[sys.argv.index('--add_measured_apendix')+1]
            mcolumns = np.array([mco + appendix for mco in mcolumns])
        mselect = get_indx(selectin, mnames)
        melect = get_indx(electin, mcolumns, islist = True)
        mnames, mvalues, mcolumns = mnames[mselect], mvalues[mselect][melect], mcolumns[melect]
        
        if barplot is not None:
            for i in range(len(mvalues)):
                barplot[i].append( mvalues[i])
        else:
            barplot = [[mv] for mv in mvalues]
        for p, pc in enumerate(mcolumns):
            print(pc, barplot[p])
        barplot = np.array(barplot)
        print(pearsonr(barplot[:,0], barplot[:,1]))
    

    vlim = None
    if '--vlim' in sys.argv:
        vlim = np.array(sys.argv[sys.argv.index('--vlim')+1].split(','), dtype = float)
    
    unit = 0.15
    if '--unit' in sys.argv:
        unit = float(sys.argv[sys.argv.index('--unit')+1])
    ratio = 10
    if '--ratio' in sys.argv:
        ratio = float(sys.argv[sys.argv.index('--ratio')+1])
    
    if '--center' in sys.argv:
        flanks = int(sys.argv[sys.argv.index('--center')+1])
        st = int((len(seq)-2*flanks)/2)
        seq = seq[st:-st]
        att = att[:,st:-st]
        newrange = [st, len(st)-st+1]
     
    if '--locations' in sys.argv:
        loc = sys.argv[sys.argv.index('--locations')+1].split(',')
        seq = seq[int(loc[0]):int(loc[1])]
        att = att[:,int(loc[0]):int(loc[1])]
        newrange = [int(loc[0]),int(loc[1])+1]
    
    dpi = 200
    if '--dpi' in sys.argv:
        dpi = int(sys.argv[sys.argv.index('--dpi')+1])
    
    fig = plot_attribution(seq, att, motifs = mlocs, seq_based = seq_based, exp = exp, vlim = vlim, unit = unit, ratio = ratio, barplot = barplot, xtick_range = newrange)
    if '--show' in sys.argv:
        plt.show()
    else:
        if '--transparent' in sys.argv:
            fig.savefig(outname+'.png', transparent = True, dpi = dpi, bbox_inches = 'tight')
        else:
            fig.savefig(outname+'.jpg', dpi = dpi, bbox_inches = 'tight')
            print(outname+'.jpg')



# algign sequences and plot attributions of both sequences

