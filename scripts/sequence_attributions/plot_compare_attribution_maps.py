import numpy as np
import sys, os
import matplotlib.pyplot as plt
import logomaker as lm
import pandas as pd
from functools import reduce



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


def logoax(fig, att, ylabel = None, ylim = None, sb = 111, pos = None, labelbottom = True, bottom = True):
    ax0 =  fig.add_subplot(sb)
    if pos not None:
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

def plot_attribution(seq, att, attb, motifs = None, seq_based = True):
        #print(att[0,:10,:,0], att[0,:10,:,0])
    ism = np.copy(att)
    ismb = np.copy(attb)
    if seq_based:
        att = -np.sum(att, axis = -1)/3
        att = seq * att[:, None]
        attb = -np.sum(attb, axis = -1)/3
        attb = seq * attb[:, None]
        ylabel = 'Attribution\nmean'
    else:
        att -= (np.sum(att, axis = -1)/3)[:,None]
        ylabel = 'Attribution'
    
    
    mina = min(0,np.amin(np.sum(np.ma.masked_greater(att,0), axis = -1)))
    maxa = np.amax(np.sum(np.ma.masked_less(att,0), axis = -1))
    minab = min(0,np.amin(np.sum(np.ma.masked_greater(attb,0), axis = -1)))
    maxab = np.amax(np.sum(np.ma.masked_less(attb,0), axis = -1))
    attlim = [min(mina,minab), max(maxab,maxa)]

    vlim = max(np.amax(np.absolute(ism)), np.amax(np.absolute(ismb)))

    fig = plt.figure(figsize = (0.15*len(seq), 3*10*0.15+3*0.15*5), dpi = 50)
    
    ax0 = logoax(fig, att, ylabel = ylabel + ' A', ylim = attlim, sb = 611, pos = [0.1,0.1+(35/45)*0.8,0.8,0.8*(10/45)], labelbottom = False, bottom = False)
    
    if motifs is not None:
        mask = motifs[:,-2] == 0
        colors = motifs[mask,-1]
        locations = [ti1[l] for l in motifs[mask,1]]
        add_frames(att, locations, colors, ax0)

    ax1 = heatax(ism, fig, pos = [0.1,0.1+(30/45)*0.8,0.8,0.8*(4/45)], sb = 612, cmap = 'coolwarm', ylabel = 'ISM A', labelbottom = False, bottom = False, vlim = vlim)
    
    ax2 = logoax(fig, attb, ylabel = ylabel + ' B', ylim = attlim, sb = 613, pos = [0.1,0.1+(20/45)*0.8,0.8,0.8*(10/45)], labelbottom = False, bottom = False)
    
    if motifs is not None:
        mask = motifs[:,-2] == 0
        colors = motifs[mask,-1]
        locations = [ti1[l] for l in motifs[mask,1]]
        add_frames(attb, locations, colors, ax2)

    ax3 = heatax(ismb, fig, pos = [0.1,0.1+(15/45)*0.8,0.8,0.8*(4/45)], sb = 614, cmap = 'coolwarm', ylabel = 'ISM B', labelbottom = False, bottom = False, vlim = vlim)
    
    ax4 = logoax(fig, att-attB, ylabel = ylabel + ' A-B', ylim = attlim, sb = 615, pos = [0.1,0.1+(5/45)*0.8,0.8,0.8*(10/45)], labelbottom = False, bottom = False)
    
    if motifs is not None:
        mask = motifs[:,-2] == 0
        colors = motifs[mask,-1]
        locations = [ti1[l] for l in motifs[mask,1]]
        add_frames(att-attB, locations, colors, ax4)

    ax5 = heatax(ism-ismb, fig, pos = [0.1,0.1+(0/45)*0.8,0.8,0.8*(4/45)], sb = 616, cmap = 'coolwarm', ylabel = 'ISM A-B', labelbottom = True, bottom = True, vlim = vlim)
    
    return fig




if __name__ == '__main__':
    ism1 = np.load(sys.argv[1], allow_pickle = True)
    ism2 = np.load(sys.argv[2], allow_pickle = True)
    ref = np.load(sys.argv[3], allow_pickle = True)
    
    select = sys.argv[4]
    elect = sys.argv[5]
    
    names1, values1, exp1 = ism1['names'], ism1['values'], ism1['experiments']
    names2, values2, exp2 = ism2['names'], ism2['values'], ism2['experiments']
    
    seqfeatures, genenames = ref['seqfeatures'], ref['genenames']
    if len(np.shape(seqfeatures)) == 1:
        seqfeatures, featurenames = seqfeatures
    
    common = reduce(np.intersect1d, [names1, names2, genenames])
    exp = np.intersect1d(exp1, exp2)
    nsort1 = np.argsort(names1)[np.isin(np.sort(names1),common)]
    nsort2 = np.argsort(names2)[np.isin(np.sort(names2),common)]
    gsort = np.argsort(genenames)[np.isin(np.sort(genenames), common)]
    values1 = values1[nsort1]
    values2 = values2[nsort2]
    esort1 = np.argsort(exp1)[np.isin(np.sort(exp1), exp)]
    values1 = values1[:,esort1]
    esort2 = np.argsort(exp2)[np.isin(np.sort(exp2), exp)]
    values2 = values2[:,esort2]
    seqfeatures = seqfeatures[gsort]
    values1 = np.transpose(values1, axes = (0,1,3,2))
    values2 = np.transpose(values2, axes = (0,1,3,2))
    outname = os.path.splitext(sys.argv[1])[0]+'-vs-'+os.path.splitext(sys.argv[2])[0]+'_'+select+'_'+elect
    if '--outname' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]
    
    if isint(select):
        select = int(select)
    else:
        select = list(common).index(select)
        
    if isint(elect):
        elect = int(elect)
    else:
        elect = list(exp).index(elect)
    
    seq = seqfeatures[select]
    att1 = values1[select, elect]
    att2 = values2[select, elect]
    print(common[select], exp[elect])
    
    mlocs = None
    if '--motif_location' in sys.argv:
        mlocfile = np.genfromtxt(sys.argv[sys.argv.index('--motif_location')+1], dtype = str)[:,0]
        mlocs = np.array([m.rsplit('_',3) for m in mlocfile])
        keep = [i for i, name in enumerate(mlocs[:,0]) if name in sys.argv[1]]
        mlocs = mlocs[keep].astype(object)
        for m, ml in enumerate(mlocs):
            mlocs[m][1] = np.array(ml[1].split('-'), dtype = int)
        mlocs[:,[2,3]] = mlocs[:,[2,3]].astype(int)

    seq_based = True
    if '--attributions' in sys.argv:
        seq_based = False

    
    fig = plot_attribution(seq, att1, att2, motifs = mlocs, seq_based = seq_based)
    if '--show' in sys.argv:
        plt.show()
    else:
        fig.savefig(outname+'.jpg', dpi = 200, bbox_inches = 'tight')



