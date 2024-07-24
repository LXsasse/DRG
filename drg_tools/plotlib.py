# plotlib.py
# functions and classes to plot data 
# Author: Alexander Sasse <alexander.sasse@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
import sys,os
from scipy.linalg import svd
from functools import reduce
from matplotlib import cm
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib as mpl
import logomaker
import pandas as pd


def _add_frames(att, locations, colors, ax):
    att = np.array(att)
    cmap = ['purple', 'limegreen']
    for l, loc in enumerate(locations):
        mina, maxa = np.amin(np.sum(np.ma.masked_greater(att[loc[0]:loc[1]+1],0),axis = 1)), np.amax(np.sum(np.ma.masked_less(att[loc[0]:loc[1]+1],0),axis = 1))
        x = [loc[0]-0.5, loc[1]+0.5]
        ax.plot(x, [mina, mina], c = cmap[colors[l]])
        ax.plot(x, [maxa, maxa], c = cmap[colors[l]])
        ax.plot([x[0], x[0]] , [mina, maxa], c = cmap[colors[l]])
        ax.plot([x[1], x[1]] , [mina, maxa], c = cmap[colors[l]])


def _logoax(fig, att, ylabel = None, ylim = None, sb = 111, pos = None, labelbottom = True, bottom = True, xticks = None, xticklabels = None):
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
    
def _heatax(ism, fig, pos = None, sb = 111, cmap = 'coolwarm', ylabel = None, labelbottom = True, bottom = True, vlim = None, yticklabels = None, xticklabels = None):
    if vlim is None:
        vlim = np.amax(np.absolute(ism))
    ax1 =fig.add_subplot(sb[0], sb[1], sb[2])
    if pos is not None:
        ax1.set_position(pos)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.imshow(ism.T, aspect = 'auto', cmap = cmap, vmin = -vlim, vmax = vlim)
    if ylabel is not None:
        ax1.set_ylabel(ylabel)
    ax1.tick_params(bottom = bottom, labelbottom = labelbottom)
    if yticklabels is not None:
        ax1.set_yticks(np.arange(len(yticklabels)))
        ax1.set_yticklabels(list(yticklabels))
    if xticklabels is not None:
        ax1.set_xticks(np.arange(len(xticklabels)))
        ax1.set_xticklabels(list(xticklabels))
    
    return ax1

def _activity_plot(values, ylim, xticklabels, ax):
    ax.bar(np.arange(len(values)), values)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(ylim)
    if xticklabels is None:
        ax.tick_params(bottom = False, labelbottom = False)
    else:
        ax.set_xticklabels(xticklabels, rotation = 60)
    return ax

def _generate_xticks(start, end, n):
    possible = np.concatenate([np.array([1,2,5,10])*10**i for i in range(-16,16)])
    steps=(end-start)/n
    steps = possible[np.argmin(np.absolute(possible - steps))]
    ticklabels = np.arange(start, end)
    ticks = np.where(ticklabels%steps == 0)[0]
    ticklabels = ticklabels[ticks]
    return ticks, ticklabels
    
    

def plot_attribution(seq, att, motifs = None, seq_based = 1, exp = None, vlim = None, unit = 0.15, ratio = 10, ylabel = None, xtick_range = None, barplot = None, heatmap = False, center_attribution = False):
    
    ism = np.copy(att)
    if center_attribution:
        att -= (np.sum(att, axis = -1)/4)[...,None]
    
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
        xticks, xticklabels = _generate_xticks(xtick_range[0], xtick_range[1], 7)
    else:
        xticks, xticklabels = None, None
    
    _heat = (1+int(heatmap))
    
    fig = plt.figure(figsize = (unit*len(seq), len(att) * _heat * ratio*unit), dpi = 50)
    
    axs = []
    for a, at in enumerate(att):
        axs.append(_logoax(fig, at, ylabel = exp[a], ylim = attlim, sb = [len(att)*_heat, 1, 1+(a*_heat)], pos = [0.1, 0.1+(len(att)-1-(a*_heat))/len(att)/_heat*0.8, 0.8, 0.8*(1/len(att)/_heat)*0.8], labelbottom = (a == len(att)-1) & (~heatmap), bottom = (a == len(att)-1)& (~heatmap), xticks = xticks, xticklabels = xticklabels))
        if heatmap:
            _vlim = np.amax(np.absolute(attlim))
            axs.append(_heatax(ism, fig, ylim = attlim, sb = [len(att)*_heat, 1, 1+(a*_heat)], pos = [0.1, 0.1+(len(att)-1-(a*_heat))/len(att)/_heat*0.8, 0.8, 0.8*(1/len(att)/_heat)*0.8], labelbottom = (a == len(att)-1), bottom = (a == len(att)-1), xticks = xticks, xticklabels = xticklabels), cmap = 'coolwarm', ylabel = None, vlim = [-_vlim, _vlim]))
            
    
    # This is for a barplot on the side of the sequence logo, that shows predicted and/or measured actibity
    if barplot is not None:
        ylim = [0, np.amax(barplot)]
        for b, bp in enumerate(barplot):
            ax = fig.add_subplot(len(barplot), len(barplot), len(barplot) + b)
            ax.set_position([0.9 + 2.5*0.8*(1/len(seq)), 0.1+(len(att)-1-b)/len(att)*0.8, 6*0.8*(1/len(seq)), 0.8*(1/len(att))*0.8])
            axs.append(_activity_plot(bp, ylim, None, ax))
    
    
    if motifs is not None:
        mask = motifs[:,-2] == 0
        colors = motifs[mask,-1]
        locations = [ti1[l] for l in motifs[mask,1]]
        _add_frames(att, locations, colors, ax0)

    return fig





def plot_single_pwm(pwm, log = False, axes = False):
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



def plot_heatmap(heatmat, # matrix that is plotted with imshow
                 ydistmat = None,
                 xdistmat = None,
                 measurex = None, # if matrix is not a symmetric distance matrix then measurex define distannce metric for linkage clustering 
                 measurey = None, # same as measurex just for y axic
                 sortx = None, # agglomerative clustering algorith used in likage, f.e average, or single
                 sorty = None, # same as above but for y axis
                 x_attributes = None, # additional heatmap with attributes of columns
                 y_attributes = None, # same as above for y axis
                 xattr_name = None, # names of attributes for columns
                 yattr_name = None, # names of attributes for rows
                 heatmapcolor = cm.BrBG_r, # color map of main matrix
                 xatt_color = None, # color map or list of colormaps for attributes
                 yatt_color = None, 
                 xatt_vlim = None, # vmin and vmas for xattributes, or list of vmin and vmax
                 yatt_vlim = None,
                 pwms = None, # pwms that are plotted with logomaker next to rows of matrix
                 infocont = True, # if True, the matrices will be plotted as information content
                 combine_cutx = 0., # NOT implemented yet, can be used to cut off linkage tree at certain distance if linkage tree too full
                 combine_cuty = 0., 
                 color_cutx = 0., # cut off for coloring in linkage tree. 
                 color_cuty = 0., 
                 xdenline = None, # line drawn into linkage tree on x-axis
                 ydenline = None, 
                 plot_value = False, # if true the values are written into the cells of the matrix
                 vmin = None, # min color value 
                 vmax = None, 
                 grid = False, # if True, grey grid drawn around heatmap cells
                 xlabel = None, # label on x-axis
                 ylabel = None, # ylabel
                 xticklabels = None,
                 yticklabels  = None,
                 showdpi = None,
                 dpi = None,
                 figname = None,
                 fmt = '.jpg',
                 maxsize = 150, 
                 cellsize = 0.3,
                 cellratio = 1.,
                 noheatmap = False,
                 row_distributions = None,
                 row_distribution_kwargs = {}):
    
    if heatmat is None:
        Nx = 0
        Ny = np.shape(ydistmat)[0]
    else:
        Ny, Nx = np.shape(heatmat)[0], np.shape(heatmat)[1]
    
    if xdistmat is None:
        xdistmat = np.copy(heatmat)
    if ydistmat is None:
        ydistmat = np.copy(heatmat)
    # either provide similarity matrix as heatmap (measurex = None) or provide a similarity function from scipy.spatial.distance.pdist
    # If no measure is provided heatmap entries will be rescaled between 0,1 and a transformation function can retransform for xticklabels
    if not noheatmap:
        if measurex is not None:
            simatrixX = pdist(xdistmat.T, metric = measurex)
        elif xdistmat is not None:
            if np.shape(xdistmat)[0] != np.shape(xdistmat)[1]:
                print( 'xdistmat not symmetric matrix: sortx set to None if given')
                sortx = None
            else:
                if np.any(np.abs(xdistmat - xdistmat.T) > 1e-8):
                    print( 'xdistmat not symmetric matrix: sortx set to None if given')
                    sortx = None
            
            if sortx is not None:        
                # checks if similarity matrix or distance matrix
                issimilarity = np.all(np.amax(xdistmat) == np.diag(xdistmat))
                heatmax, heatmin = np.amax(xdistmat), np.amin(xdistmat)
                simatrixX = int(issimilarity) - (2.*int(issimilarity)-1.) * (xdistmat - heatmin)/(heatmax - heatmin)
                simatrixX = simatrixX[np.triu_indices(len(simatrixX),1)]
        else:
            sortx = None
            simatrixX = None
                
    if measurey is not None:
        simatrixY = pdist(ydistmat, metric = measurey)
    elif ydistmat is not None:
        if np.shape(ydistmat)[0] != np.shape(ydistmat)[1]:
            print( 'ydistmat not symmetric matrix: sorty set to None if given')
            sorty = None
        else:
            if np.any(np.abs(ydistmat - ydistmat.T) > 1e-8):
                print( 'ydistmat not symmetric matrix: sorty set to None if given')
                sorty = None
        if sorty is not None:        
            # checks if similarity matrix or distance matrix
            issimilarity = np.all(np.amax(ydistmat) == np.diag(ydistmat))
            heatmax, heatmin = np.amax(ydistmat), np.amin(ydistmat)
            simatrixY = int(issimilarity) - (2.*int(issimilarity)-1.) * (ydistmat - heatmin)/(heatmax - heatmin)
            simatrixY = simatrixY[np.triu_indices(len(simatrixY),1)]
    else:
        sorty = None
        simatrixY = None
            
    
    
    # Generate dendrogram for x and y
    #### NJ not yet included
    if sortx is not None and not noheatmap:
        Zx = linkage(simatrixX, sortx)
        #if combine_cutx > 0:
            #Zx = reduce_z(Zx, combine_cutx)
    
    if sorty is not None:
        Zy = linkage(simatrixY, sorty) 
        #if combine_cuty > 0:
            #Zy = reduce_z(Zy, combine_cuty)
    
    if not noheatmap and heatmat is not None:
        if cellsize*np.shape(heatmat)[1] > maxsize:
            xticklabels = None
            plot_value = False
            yattr_name = None
            cellsize = min(maxsize/(np.shape(heatmat)[0]*cellratio), maxsize/np.shape(heatmat)[1])
    
        if cellsize*np.shape(heatmat)[0] *cellratio > maxsize:
            yticklabels = None
            plot_value = False
            x_attr_name = None
            cellsize = min(maxsize/(np.shape(heatmat)[0]*cellratio), maxsize/np.shape(heatmat)[1])
    
    xextra = 0.
    if y_attributes is not None:
        xextra = np.shape(y_attributes)[1] + 0.25
    yextra = 0.
    if x_attributes is not None:
        yextra = np.shape(x_attributes)[0] + 0.25
    denx, deny, pwmsize, rowdistsize = 0, 0, 0, 0
    if sortx is not None and not noheatmap:
        denx = 10 + 0.25
    if sorty is not None:
        deny = 3+.25
    if pwms is not None:
        pwmsize = 3.25
    if row_distributions is not None:
        rowdistsize = 6+ 0.25
    
    basesize = 0
    
    wfig = cellsize*(Nx+xextra+deny+pwmsize+rowdistsize+basesize)
    hfig = cellsize*cellratio*(Ny+yextra/cellratio+denx+basesize)
    
    #print(Nx,xextra,deny,pwmsize,rowdistsize,basesize)
    #print(Ny,yextra/cellratio,denx,basesize)
    #print(wfig, hfig)
    fig = plt.figure(figsize = (wfig, hfig), dpi = showdpi)
    
    fullw = Nx+xextra+deny+pwmsize+rowdistsize+basesize
    fullh = Ny+yextra+denx+basesize
    
    left = 0.1
    bottom = 0.1
    width = 0.8
    height = 0.8
    before = width*(deny+pwmsize)/fullw
    after = width*(xextra+rowdistsize)/fullw
    beneath = height*yextra/fullh
    above = height*denx/fullh
    
    wfac = width * Nx/fullw
    mfac = height * Ny/fullh
    #print(wfac, mfac)
    
    if not noheatmap:
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_position([0.1+before,0.1+beneath, wfac, mfac])
        ax.tick_params(which = 'both', bottom = False, labelbottom = False, left = False, labelleft = False)
    
    # plot dendrogram for x axis
    if sortx is not None and not noheatmap:
        axdenx = fig.add_subplot(711)
        axdenx.spines['top'].set_visible(False)
        axdenx.spines['right'].set_visible(False)
        axdenx.spines['bottom'].set_visible(False)
        axdenx.tick_params(which = 'both', bottom = False, labelbottom = False)
        axdenx.set_position([0.1+before,0.9 - above, wfac, height*(denx-0.25)/fullh])
        dnx = dendrogram(Zx, ax = axdenx, no_labels = True, above_threshold_color = 'k', color_threshold = color_cutx, orientation = 'top')
        
        sortx = dnx['leaves']
        heatmat = heatmat[:, sortx]
        if x_attributes is not None:
            x_attributes = x_attributes[:, sortx]
            
        if xticklabels is not None:
            xticklabels = xticklabels[sortx]
            
        if xdenline is not None:
            axdenx.plot([0,len(heatmat[0])*10], [xdenline, xdenline], color = 'r')
    elif heatmat is not None:
        sortx = np.arange(Nx, dtype = int)
    
    sys.setrecursionlimit(100000)    
    
    
    if sorty is not None:
        axdeny = fig.add_subplot(171)
        axdeny.spines['top'].set_visible(False)
        axdeny.spines['right'].set_visible(False)
        axdeny.spines['left'].set_visible(False)
        axdeny.tick_params(which = 'both', left = False, labelleft = False)
        axdeny.set_position([0.1,0.1+beneath, width*(deny-0.25)/fullw, mfac])
        dny = dendrogram(Zy, ax = axdeny, no_labels = True, color_threshold = color_cuty, above_threshold_color = 'k', orientation = 'left', get_leaves = True)
        sorty = dny['leaves']
        if heatmat is not None:
            heatmat = heatmat[sorty]
        #axdeny.set_yticks(axdeny.get_yticks()[1:])

        if y_attributes is not None:
            y_attributes = y_attributes[sorty]
            
        if yticklabels is not None:
            #print(yticklabels)
            yticklabels = yticklabels[sorty]
            #print(sorty, yticklabels)
        if ydenline is not None:
            axdeny.plot([ydenline, ydenline], [0,len(heatmat)*10], color = 'r')
    elif heatmat is not None:
        sorty = np.arange(len(heatmat), dtype = int)
    
    if pwms is not None:
        if infocont:
            pwm_min, pwm_max = 0,2
        else:
            pwm_min, pwm_max = 0, int(np.ceil(np.amax([np.amax(pwm) for pwm in pwms])))
        for s, si in enumerate(sorty[::-1]):
            axpwm = fig.add_subplot(len(sorty),1,s+1)
            axpwm.set_position([0.1+before-pwmsize*width/fullw, 0.1+beneath+mfac-mfac*(s+0.9)/len(sorty), (pwmsize-0.25)*width/fullw, mfac/len(sorty) *0.8])
            #axpwm.set_position([0.2-mfac*3.125*cellsize/wfig, 0.2+mfac-mfac*(s+.9)/len(sorty),mfac*3*cellsize/wfig, mfac*0.8*cellsize/hfig])
            pwm = pwms[si]
            if infocont:
                pwm = np.log2((pwms[si]+1e-16)/0.25)
                pwm[pwm<0] = 0
            logomaker.Logo(pd.DataFrame(pwm, columns = list('ACGT')), ax = axpwm, color_scheme = 'classic')
            axpwm.set_ylim([pwm_min, pwm_max])
            
            axpwm.spines['top'].set_visible(False)
            axpwm.spines['right'].set_visible(False)
            axpwm.spines['left'].set_visible(False)
            axpwm.tick_params(labelleft = False, labelbottom = False, bottom = False)
            if noheatmap and row_distributions is None and yticklabels is not None:
                axpwm.tick_params(labelleft = False, labelright = True, labelbottom = False, bottom = False)
                axpwm.set_yticks([(pwm_max+pwm_min)/2])
                axpwm.set_yticklabels(yticklabels[[-s-1]])
                
    if not noheatmap:
        if vmin is None:
            vmin = np.amin(heatmat)
        if vmax is None:
            vmax = np.amax(heatmat)
        
        ax.imshow(heatmat, aspect = 'auto', cmap = heatmapcolor, vmin = vmin, vmax = vmax, origin = 'lower')
        ax.set_yticks(np.arange(len(heatmat)))
        ax.set_xticks(np.arange(len(heatmat[0])))
        
        
        # colorbar
        axcol = fig.add_subplot(999)  
        print(vmin, vmax)
        axcol.set_position([0.1+before+wfac+ width*0.25/fullw, 0.1+beneath+mfac+ height*0.25/fullh, width*5/fullw, height*1/fullh])
        axcol.tick_params(bottom = False, labelbottom = False, labeltop = True, top = True, left = False, labelleft = False)
        axcol.imshow(np.linspace(0,1,101).reshape(1,-1), aspect = 'auto', cmap = heatmapcolor)
        axcol.set_xticks([0,101])
        heatmapresolution = 1
        heatmapresolution = ['Repressive', 'Activating']
        if isinstance(heatmapresolution, int):
            axcol.set_xticklabels([round(vmin,heatmapresolution), round(vmax,heatmapresolution)], rotation = 60)
        elif isinstance(heatmapresolution,list):
            axcol.set_xticklabels([heatmapresolution[0], heatmapresolution[-1]], rotation = 60)
            
        
        
        if plot_value:
            if np.amax(np.absolute(heatmat)) > 10:
                heattext = np.array(heatmat, dtype = int)
            elif np.amax(np.absolute(heatmat)) > 1:
                heattext = np.around(heatmat, 1)
            else:
                heattext = np.around(heatmat, 2)
            for c in range(len(heattext[0])):
                for d in range(len(heattext)):
                    ax.text(c,d,str(heattext[d,c]), color = 'k', ha = 'center', fontsize = 6)
        
        
        if grid:
            ax.set_yticks(np.arange(len(heatmat)+1)-0.5, minor = True)
            ax.set_xticks(np.arange(len(heatmat[0])+1)-0.5, minor = True)
            ax.grid(color = 'k', which = 'minor')


        if x_attributes is not None and not noheatmap:
            for x, xunique in enumerate(x_attributes):
                xunique = np.unique(xunique)
                for s, xuni in enumerate(xunique):
                    x_attributes[x, x_attributes[x] == xuni] = s
            x_attributes = x_attributes.astype(int)
            axatx = fig.add_subplot(717)
            axatx.spines['top'].set_visible(False)
            axatx.spines['bottom'].set_visible(False)
            axatx.spines['right'].set_visible(False)
            axatx.spines['left'].set_visible(False)
            axatx.tick_params(which = 'both', bottom = False, labelbottom = False, left = False, labelleft = False, labelright = False)
            #axatx.set_position([0.2, 0.2-mfac*(np.shape(x_attributes)[0]+0.25)*cellsize/hfig,mfac, mfac*np.shape(x_attributes)[0]*cellsize/hfig])
            axatx.set_position([0.1+before,0.1, wfac, height*(yextra-0.25)/fullh])
            if isinstance(xatt_color,list):
                for xai, xac in enumerate(xatt_color):
                    mask = np.ones(np.shape(x_attributes))
                    mask[xai] = 0
                    mask = mask == 1
                    axatx.imshow(np.ma.masked_array(x_attributes,mask), aspect = 'auto', cmap = xac, vmin = np.amin(x_attributes[xai]), vmax = np.amax(x_attributes[xai]))
            else:
                axatx.imshow(x_attributes, aspect = 'auto', cmap = xatt_color)
            axatx.set_xticks(np.arange(len(heatmat[0])))        
            if xattr_name is not None:
                axatx.tick_params(labelright = True)
                axatx.set_yticks(np.arange(np.shape(x_attributes)[0]))
                axatx.set_yticklabels(xattr_name)
            
            if xlabel is not None:
                axatx.tick_params(which = 'both', bottom = False, labelbottom = True, left = False, labelleft = False)
                axatx.set_xlabel(xlabel)
            if xticklabels is not None:
                axatx.tick_params(which = 'both', bottom = True, labelbottom = True, left = False, labelleft = False)
                axatx.set_xticklabels(xticklabels, rotation  = 90)
        
        elif xlabel is not None and not noheatmap:
            ax.set_xlabel(xlabel)
        elif xticklabels is not None and not noheatmap:
            ax.tick_params(which = 'both', bottom = True, labelbottom = True, left = False, labelleft = False)
            ax.set_xticklabels(xticklabels, rotation = 90)
                
        
        if y_attributes is not None:
            #print(y_attributes)
            for y, yunique in enumerate(y_attributes.T):
                if not (isinstance(yunique[0],int) or isinstance(yunique[0], float)):
                    yunique = np.unique(yunique)
                    for s, yuni in enumerate(yunique):
                        y_attributes[y_attributes[:,y] == yuni,y] = s
            y_attributes = y_attributes.astype(float)
            #print(y_attributes)
            axaty = fig.add_subplot(177)
            axaty.spines['top'].set_visible(False)
            axaty.spines['bottom'].set_visible(False)
            axaty.spines['right'].set_visible(False)
            axaty.spines['left'].set_visible(False)
            axaty.tick_params(which = 'both', bottom = False, labelbottom = False, left = False, labelleft = False)
            #axaty.set_position([0.2+wfac+mfac*0.25*cellsize/wfig,0.2,mfac*np.shape(y_attributes)[1]*cellsize/wfig,mfac])
            axaty.set_position([0.1+before+wfac,0.1+beneath, width*(xextra-0.25)/fullw, mfac])
            if isinstance(yatt_color,list):
                if not (isinstance(yatt_vlim, list) and len(yatt_vlim) == len(yatt_color)):
                    yatt_vlim = [[None, None] for i in range(len(yatt_color))]
                for yai, yac in enumerate(yatt_color):
                    mask = np.ones(np.shape(y_attributes))
                    mask[:,yai] = 0
                    mask = mask == 1
                    axaty.imshow(np.ma.masked_array(y_attributes,mask), aspect = 'auto', cmap = yac, vmin = yatt_vlim[yai][0], vmax =yatt_vlim[yai][1],origin = 'lower')
            else:
                axaty.imshow(y_attributes, aspect = 'auto', cmap = yatt_color, origin = 'lower')
            if yattr_name is not None:
                axaty.tick_params(labeltop = True)
                axaty.set_xticks(np.arange(np.shape(y_attributes)[1]))
                axaty.set_xticklabels(yattr_name, rotation = 270)
            
            axaty.set_yticks(np.arange(len(heatmat)))
            if ylabel is not None:
                axaty.tick_params(labelright = True)
                axaty.set_ylabel(ylabel)
            if yticklabels is not None:
                axaty.tick_params(labelright = True, right = True)
                #print('int', yticklabels)
                axaty.set_yticklabels(yticklabels)
        
        elif ylabel is not None and not noheatmap:
            ax.set_ylabel(ylabel)
        elif yticklabels is not None and not noheatmap:
            #print(yticklabels)
            ax.tick_params(which = 'both', bottom = False, labelbottom = True, left = False, labelleft = False, labelright = True)
            ax.set_yticklabels(yticklabels)
        
    if row_distributions is not None:
        if not isinstance(row_distributions, list) and not isinstance(row_distributions, np.ndarray):
            row_distributions = list(matrix)
        axdy = fig.add_subplot(188)
        axdy.spines['top'].set_visible(False)
        axdy.spines['right'].set_visible(False)
        axdy.tick_params(which = 'both', left = False, labelleft = False, labelright = True, right = True)
        
        if y_attributes is not None:
            dwidth = mfac*np.shape(y_attributes)[1]*cellsize/wfig++mfac*0.25*cellsize/wfig
        else:
            dwidth = 0
        #axdy.set_position([0.2+wfac+mfac*0.25*cellsize/wfig + dwidth,0.2, mfac*8*cellsize/wfig, mfac])
        axdy.set_position([0.1+before+wfac+width*(xextra-0.25)/fullw, 0.1+beneath, width*(rowdistsize-0.25)/fullw, mfac])
        if sorty is not None:
            yticklabels = yticklabels[np.argsort(sorty)]
        plot_distribution(row_distributions, yticklabels, vert = False, labelside = 'opposite', ax = axdy, sort = sorty, outname = None, **row_distribution_kwargs)
    

    if figname is not None:
        if not noheatmap:
            figname += '_heatmap'
        fig.savefig(figname+fmt, dpi = dpi, bbox_inches = 'tight')
        print( 'SAVED', figname+fmt, dpi)
    else:
        plt.show()
    plt.close()
    return sortx, sorty


def simple_beeswarm(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x

def approximate_density(set1, bins = 20, moves = 4, miny=None, maxy = None):
    if miny is None:
        miny = np.amin(set1)
    if maxy is None:
        maxy = np.amax(set1)
    bsize = (maxy-miny)/bins
    for m in range(moves):
        bins1 = np.linspace(miny-(m+1)*bsize/(moves+1), maxy-(m+1)*bsize/(moves+1),bins + 1)
        bins2 = np.linspace(miny+(m+1)*bsize/(moves+1), maxy+(m+1)*bsize/(moves+1),bins + 1)
        density= np.array([np.sum((set1 >= bins1[b]) * (set1<bins1[b+1])) for b in range(len(bins1)-1)])
        density2= np.array([np.sum((set1 >= bins2[b]) * (set1<bins2[b+1])) for b in range(len(bins2)-1)])
        density = density/np.amax(density)
        density2 = density2/np.amax(density2)
        dens = np.zeros(len(set1))
        dcount = np.zeros(len(set1))
        for b in range(bins):
            dens[(set1 >= bins1[b]) * (set1<bins1[b+1])] += density[b]
            dens[(set1 >= bins2[b]) * (set1<bins2[b+1])] += density[b]
            dcount[(set1 >= bins1[b]) * (set1<bins1[b+1])] += 1
            dcount[(set1 >= bins2[b]) * (set1<bins2[b+1])] += 1
        dens = dens/dcount
    return dens

def plot_distribution(
    matrix, 
    modnames, 
    vert = True, 
    labelside = 'default', 
    ax = None, 
    sort = None, 
    split = 1, 
    legend_labels = None,
    legend_above = True,
    outname = None, 
    xwidth = 0.6, 
    height = 4, 
    width = 0.8, 
    show_mean = False, 
    showfliers = False, 
    showcaps = True, 
    facecolor = None, 
    mediancolor = None, 
    grid = True, 
    swarm = False,
    barplot = False, 
    plotnames = 0, 
    datanames = None, 
    scatter_color = 'grey', 
    scatter_colormap = cm.jet, 
    scatter_alpha = 0.8, 
    scatter_size = 0.5, 
    connect_swarm = False, 
    scattersort = 'top', 
    ylim = None, 
    sizemax = 2, 
    sizemin = 0.25, 
    colormin = None, 
    colormax = None, 
    dpi = 200, 
    savedpi = 200, 
    xorder = 'size', 
    ylabel = None, 
    fmt = 'jpg'):
    
    if sort is not None:
        positions = np.argsort(sort)
        modnames = np.array(modnames)[sort]
        #for m, mn in enumerate(modnames):
            #print(mn)
            #print(np.mean(matrix[sort[m]]))
            #print(sort[m], np.where(positions == m)[0], np.mean(matrix[np.where(positions == m)[0][0]]))
    else:
        positions = np.arange(len(matrix))
    
    fcolor = None
    if split > 1:
        if len(matrix) == split:
            matrix = [m for ma in matrix for m in ma]
        
        if width * split >1:
            width = width/split
        
        positions = []
        for s in range(split):
            if sort is None:
                positions.append(np.arange(int(len(matrix)/split)) + width*s - (split*width/2) + width/2)
            else:
                positions.append(np.argsort(sort) + width*s - (split*width/2) + width/2)
        positions = np.concatenate(positions)
        
        if isinstance(facecolor, list):
            if len(facecolor) == split:
                fcolor = [facecolor[c] for c in range(split) for j in range(int(len(matrix)/split))]
            else:
                fcolor = [facecolor[c] for c in range(len(matrix))]
            #facecolor = None
        
        if mediancolor is not None:
            if isinstance(mediancolor, list):
                if len(mediancolor) == split:
                    mediancolor = [mediancolor[c] for c in range(split) for j in range(int(len(matrix)/split))]
                else:
                    mediancolor = [mediancolor[c] for c in range(len(matrix))]
            else:
                mediancolor = [mediancolor for c in range(len(matrix))]
            

    if mediancolor is not None:
        if not isinstance(mediancolor, list):
            mediancolor = [mediancolor for mc in range(len(matrix))]
    
    return_ax = False
    if ax is None:
        if vert:
            fig = plt.figure(figsize = (len(modnames)*xwidth, height), dpi = dpi)
        else:
            fig = plt.figure(figsize = (height, len(modnames)*xwidth), dpi = dpi)
        ax = fig.add_subplot(111)
        ax.set_position([0.1,0.1,0.8,0.8])
    else: 
        return_ax = True
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    matrix = list(matrix)
    
    if swarm:
        #sns.swarmplot(data=matrix, color = 'k', size = size, ax = ax)
        # replace swarmplot with scatterplot with random location within width
        
        if colormin is None and not isinstance(scatter_color, str):
            colormin = np.amin(scatter_color)
        if colormax is None and not isinstance(scatter_color, str):
            colormax = np.amax(scatter_color)
        
        if connect_swarm and len(np.shape(matrix)) > 1:
            xposses = []
            randomshift = np.random.random(len(matrix[0]))
        
        for i, set1 in enumerate(matrix):
            set1 = np.array(set1)
            if scattersort == 'top':
                setsort = np.argsort(set1)
            else:
                setsort = np.argsort(-set1)
            
            if scatter_color is None:
                scatter_colormap = cm.twilight
                sccolor = np.ones(len(setsort))*0.25
            elif isinstance(scatter_color, str):
                sccolor = np.array([scatter_color for ci in range(len(setsort))])
            else:
                sccolor = (scatter_color-colormin)/(colormax-colormin)
                
            if scatter_size is None:
                scsize = 0.2*np.ones(len(setsort))*plt.rcParams['lines.markersize'] ** 2.
            elif isinstance(scatter_size, float) or isinstance(scatter_size, int):
                scsize = scatter_size * np.ones(len(setsort))*plt.rcParams['lines.markersize'] ** 2.
            else:
                scsize = np.sqrt(scatter_size/3.)
                scsize = (((sizemax-sizemin)*(scsize - np.amin(scsize))/(np.amax(scsize) - np.amin(scsize))) + sizemin)
                scsize *= plt.rcParams['lines.markersize'] ** 2.
                
            if xorder == 'size' and scatter_size is not None and not isinstance(scatter_size, float) and not isinstance(scatter_size, int):
                randx = positions[i] + width * ((scsize-np.amin(scsize))/(np.amax(scsize)-np.amin(scsize)) - 0.5)
            else:
                dens = approximate_density(set1)
                if connect_swarm and len(np.shape(matrix)) > 1:
                    randx = positions[i] + dens *width/2 * (randomshift-0.5)
                else:
                    randx = positions[i] + dens * width * (np.random.random(len(setsort))-0.5) # + width/2 * simple_beeswarm(set1, nbins = 40) #
            if vert: 
                ax.scatter(randx[setsort], set1[setsort], cmap= scatter_colormap, s = scsize[setsort], c = sccolor[setsort], alpha = scatter_alpha, vmin = 0, vmax = 1, lw = 0, zorder = 5)
            else:
                ax.scatter(set1[setsort], randx[setsort], cmap= scatter_colormap, s = scsize[setsort], c = sccolor[setsort], alpha = scatter_alpha, vmin = 0, vmax = 1, lw = 0, zorder = 5)
            if connect_swarm and len(np.shape(matrix)) > 1:
                xposses.append(randx)
        
        if connect_swarm and len(np.shape(matrix)) > 1:
            xposses=np.array(xposses)
            for j, setj in enumerate(np.array(matrix).T):
                if vert:
                    ax.plot(xposses[:,j], setj, color  = 'grey', alpha = 0.5, lw = 0.5)
                else:
                    ax.plot(setj, xposses[:,j], color  = 'grey', alpha = 0.5, lw = 0.5)
            
            
        # Determine which scatters should get a name written next to them
        if plotnames> 0 and datanames is not None:
            for mat in matrix:
                which_to_plot = np.argsort(-mat)[:plotnames]
    # generate colorbar
    if ((scatter_color is not None) and (not isinstance(scatter_color, str))) and swarm:
        axcol = fig.add_subplot(911)
        axcol.set_position([0.6,0.925,0.3, 0.05])
        axcol.tick_params(bottom = False, labelbottom = False, labeltop = True, top = True, left = False, labelleft = False)
        axcol.imshow(np.linspace(0,1,101).reshape(1,-1), aspect = 'auto', cmap = scatter_colormap)
        axcol.set_xticks([0,50,101])
        axcol.set_xticklabels([round(colormin,1), round((colormin+colormax)/2,1), round(colormax,1)], rotation = 90)
    
    if barplot:
        if fcolor is not None:
            barcolor = fcolor
        else:
            barcolor = 'grey'
        if vert:
            bplot = ax.bar(positions, np.mean(matrix,axis = 1), width = width*0.9, color = barcolor, linewidth = 1)
        else:
            if len(np.shape(matrix)) > 1:
                matrix = np.mean(matrix, axis = 1)
            bplot = ax.barh(positions, matrix, height = width*0.9, color = barcolor, linewidth = 1)
            ax.set_ylim([np.amin(positions)-0.5, np.amax(positions)+0.5])
        # create a legend()
        
        if isinstance(facecolor, list) and legend_labels is not None:
            handles = []
            for f, fcol in enumerate(facecolor):
                patch = mpatches.Patch(color=fcol, label=legend_labels[f])
                handles.append(patch)
            ax.legend(handles = handles)
    else:
        if facecolor is None or fcolor is not None:
            boxplotcolor = (0,0,0,0)
        else:
            boxplotcolor = facelolor
        
        bplot = ax.boxplot(matrix, positions = positions, vert = vert, showcaps=showcaps, patch_artist = True, boxprops={'facecolor':boxplotcolor}, showfliers=showfliers, whiskerprops={'linewidth':1}, widths = width,zorder = 4)
    
        if fcolor is not None:
            for patch, color in zip(bplot['boxes'], fcolor):
                patch.set_facecolor(color)
                fc = patch.get_facecolor()
                patch.set_facecolor(mpl.colors.to_rgba(fc, 0.7))
            # create legend()
            if isinstance(facecolor, list) and legend_labels is not None:
                handles = []
                for f, fcol in enumerate(facecolor):
                    patch = mpatches.Patch(color=fcol, label=legend_labels[f])
                    handles.append(patch)
                if legend_above:
                    ax.legend(handles = handles,bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=3)
                else:
                    ax.legend(handles = handles)
                
                
        if mediancolor is not None:
            for mx, median in enumerate(bplot['medians']):
                median.set_color(mediancolor[mx])
    
    if ylim is not None:
        if vert:
            ax.set_ylim(ylim)
        else:
            ax.set_xlim(ylim)
    
    if show_mean:
        if vert:
            ax.plot(np.sort(positions), [np.mean(matrix[s]) for s in np.argsort(positions)], color = 'r', marker = 's', zorder = 6, markersize = 3, ls = '--', lw = 0.5)
        else:
            ax.plot([np.mean(matrix[s]) for s in np.argsort(positions)],np.sort(positions), color = 'r', marker = 's', zorder = 6, markersize = 3, ls = '--', lw = 0.5)
    
    if vert:
        if labelside =='opposite':
            ax.tick_params(bottom = False, labelbottom = False, labeltop = True)
        ax.set_xticks(np.arange(len(modnames)))
        ax.set_xticklabels(modnames, rotation = 90)
        if grid:
            ax.grid(axis = 'y')
    else:
        if labelside =='opposite':
            ax.tick_params(left = False, labelleft = False, labelright = True)
        ax.set_yticks(np.arange(len(modnames)))
        ax.set_yticklabels(modnames)
        if grid:
            ax.grid(axis = 'x')
    
    if return_ax:
        return ax
    
    if outname is None:
        #fig.tight_layout()
        plt.show()
    else:
        fig.savefig(outname+'_distribution.'+fmt, dpi = savedpi, bbox_inches = 'tight')




def vulcano(fc, pv, figname, xcutoff=1., ycutoff = 1.96, text = None, mirror = False, eigenvalues = False, colors = None, cmap = 'viridis'):
    fig = plt.figure(figname, figsize = (4,4), dpi = 200)
    ax = fig.add_subplot(111)
    maskx = np.absolute(fc) > xcutoff
    masky = pv > ycutoff
    if mirror:
        pv = np.sign(fc) * pv
    if colors is not None:
        sort = np.argsort(colors)
        ap = ax.scatter(fc[sort], pv[sort], cmap = cmap, c = colors[sort], alpha = 0.6)
        fig.colorbar(ap, aspect = 2, pad = 0, anchor = (0,0.9), shrink = 0.15)
    else:
        ax.scatter(fc[~(maskx*masky)], pv[~(maskx*masky)], c = 'grey', alpha = 0.3)
        ax.scatter(fc[maskx*masky], pv[maskx*masky], c='maroon', alpha = 0.8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(figname)
    ax.set_xlabel('Log2 Fold change')
    ax.set_ylabel('Log10 p-value')
    if mirror:
        ax.set_ylabel('Log10 sign p-value')
    if eigenvalues:
        u, s, v = svd(np.array([fc,pv]), full_matrices=False, compute_uv=True)
        u = u/u[0]
        ax.plot([-u[0,0],0, u[0,0]], [-u[1,0], 0, u[1,0]], color = 'r', ls = '--', label = 'Eigval1 (m='+ str(np.around(u[1,0]/u[0,0],2))+ ')')
        ax.legend()
        
    
    if text is not None:
        for s in np.where(masky*maskx)[0]:
            if fc[s] < 0:
                ax.text(fc[s], pv[s], text[s], ha = 'left', size = 8)
            else:
                ax.text(fc[s], pv[s], text[s],ha = 'right', size = 8)
    fig.tight_layout()
    return fig




def plotHist(x, y = None, xcolor='navy', yaxis = False, xalpha= 0.5, ycolor = 'indigo', yalpha = 0.5, addcumulative = False, bins = None, xlabel = None, title = None, logx = False, logy = False, logdata = False):
    fig = plt.figure(figsize = (3.5,3.5))
    axp = fig.add_subplot(111)
    axp.spines['top'].set_visible(False)
    axp.spines['right'].set_visible(False)
    
    if logdata:
        x = np.log10(x+1)
    
    a,b,c = axp.hist(x, bins = bins, color = xcolor, alpha = xalpha)
    if y is not None:
        ay,by,cy = axp.hist(y, bins = bins, color = ycolor, alpha = yalpha)
    
    if addcumulative != False:
        axp2 = axp.twinx()
        axp2.spines['top'].set_visible(False)
        axp2.spines['left'].set_visible(False)
        axp2.tick_params(bottom = False, labelbottom = False)
        axp2.set_yticks([0.25,0.5,0.75,1])
        axp2.set_yticklabels([25,50,75,100])
        if addcumulative == 2:
            addcumulative = 1
            ag_,bg_,cg_ = axp2.hist(x, color = 'maroon', alpha = 1, density = True, bins = bins, cumulative = -1, histtype = 'step')
        ag,bg,cg = axp2.hist(x, color = xcolor, alpha = 1, density = True, bins = bins, cumulative = addcumulative, histtype = 'step')
        if y is not None:
            agy,bgy,cgy = axp2.hist(y, color = ycolor, alpha = 1, density = True, bins = bins, cumulative = addcumulative, histtype = 'step')
    
    
    
    if yaxis:
        print('yaxis',np.amax(a))
        axp.plot([0,0], [0, np.amax(a)], c = 'k', zorder = 5)
    
    if logx:
        if addcumulative:
            axp2.set_xscale('symlog')
        axp.set_xscale('symlog')
        
    if logy:
        axp.set_yscale('symlog')
    
    if xlabel is not None:
        axp.set_xlabel(xlabel)
    if title is not None:
        axp.set_title(title)
    return fig




def scatter3D(x,y,z, axis = True, color = None, cmap = None, xlabel = None, ylabel = None, zlabel=None, alpha = 0.9, diag = False):
    fig = plt.figure(figsize = (4,4), dpi = 150)
    ax = fig.add_subplot(111, projection='3d') #plt.axes(projection='3d')
    xlim = np.array([np.amin(x), np.amax(x)])
    ylim = np.array([np.amin(y), np.amax(y)])
    zlim = np.array([np.amin(z), np.amax(z)])
    lrat = 0.5
    if axis:
        # plot axis in there
        xlim0, ylim0, zlim0 = xlim * lrat, ylim * lrat, zlim * lrat
        ax.plot3D(xlim0,[0,0],[0,0], color = 'k', lw = 1)
        ax.plot3D([0,0],ylim0,[0,0], color = 'k', lw = 1)
        ax.plot3D([0,0],[0,0],zlim0, color = 'k', lw = 1)
    if diag:
        maxlim = np.array([xlim, ylim,zlim])*lrat
        print(maxlim)
        maxlim = [np.amax(maxlim[:,0]), np.amin(maxlim[:,1])]
        print(maxlim)
        ax.plot3D(maxlim, maxlim, maxlim, color = 'maroon', lw = 1)
    # plot a scatterplot in 3d
    ax.scatter3D(x, y, z, c=color, cmap=cmap, lw=0, alpha = alpha, s = 3)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    if zlabel is not None:
        ax.set_zlabel(zlabel)
    lrat = 0.75
    ax.set_xlim(xlim*lrat)
    ax.set_xlim(ylim*lrat)
    ax.set_xlim(zlim*lrat)
    ax.view_init(elev=25, azim=-49)
    return fig

def plot_bars(x, width = 0.8, xticklabels=None, xlabel = None, ylabel=None, 
        ylim=None, color = None, figsize = (3.5,3.5), labels = None, 
        title = None, horizontal = False):
    
    """
    Parameter
    ---------
    x : list or np.array, shape = (n_bars,) or (n_bars, n_models), or 
    (n_bars, n_models, n_conditions)
    
    Return
    ------
    
    fig : matplotlib.pyplot.fig object
    
    """

    x = np.array(x)
    positions = np.arange(np.shape(x)[0])
    xticks = np.copy(positions)
    
    if len(np.shape(x)) >1:
        n_models = np.shape(x)[1]
        bst = -width/2
        width = width/n_models
        shifts = [bst + width/2 + width *n for n in range(n_models)]
        positions = [positions+shift for shift in shifts]
        if color is None:
            color = [None for i in range(np.shape(x)[1])]
   
    if len(np.shape(x)) > 2:
        if horizontal: 
            fig = plt.figure(figsize = (figsize[0]* np.shape(x)[-1], 
                figsize[1]))
        else:
            fig = plt.figure(figsize = (figsize[0], 
                figsize[1] * np.shape(x)[-1]))
        
        for a in range(np.shape(x)[-1]):
            if horizontal:
                ax = fig.add_subplot(1, np.shape(x)[-1], a+1)
            else:
                ax = fig.add_subplot(np.shape(x)[-1], 1, a+1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
            for p, pos in enumerate(positions):
                if labels is None:
                    plotlabel = None
                else:
                    plotlabel = labels[p]
                ax.bar(pos, x[:,p,a],  width = width, color = color[p], 
                        label = plotlabel)
            
            if labels is not None:
                ax.legend()
        
            ax.grid()
            
            if horizontal: 
                if a == 0:
                    if ylabel is not None:
                        ax.set_ylabel(ylabel)    
                if xticklabels is not None:
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels, rotation = 60)
                if xlabel is not None:
                    ax.set_xlabel(xlabel)
                
            else:
                if a + 1 == np.shape(x)[-1]:
                    if xticklabels is not None:
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(xticklabels, rotation = 60)
                    if xlabel is not None:
                        ax.set_xlabel(xlabel)
                else:
                    ax.tick_params(labelbottom = False)
            
                if ylabel is not None:
                    ax.set_ylabel(ylabel)
            
            if ylim is not None:
                ax.set_ylim(ylim)
            
            if title is not None:
                ax.set_title(title[a])
        
        if horizontal:
            plt.subplots_adjust(wspace=0.2)
        else:
            plt.subplots_adjust(hspace=0.2)
    else:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if len(np.shape(x))>1:
            for p, pos in enumerate(positions):
                if labels is None:
                    plotlabel = None
                else:
                    plotlabel = labels[p]
                ax.bar(pos, x[:,p], width = width, color = color[p], 
                        label = plotlabel)
        else:
            ax.bar(positions, x, width = width, color = color, label = label)
        if labels is not None:
            ax.legend()
    
        ax.grid()
        
        if xticklabels is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation = 60)
        
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        
        if ylim is not None:
            ax.set_ylim(ylim)
    
    return fig








def plot_scatter(Ytest, Ypred, titles = None, xlabel = None, ylabel = None, outname = None, include_lr = True, include_mainvar = True, color=None, color_density = False, size = None, alpha = None, lw = None):
    n = len(Ytest[0])
    if n > 100:
        print('Number of examples is too large', n)
        return
    x_col = int(np.sqrt(n))
    y_row = int(n/x_col) + int(n%x_col!= 0)
    fig = plt.figure(figsize = (x_col*3.5,y_row*3.5), dpi = 100)
    for e in range(n):
        ax = fig.add_subplot(y_row, x_col, e+1)
        pcorr = pearsonr(Ytest[:,e], Ypred[:,e])[0]
        if titles is not None:
            ax.set_title(titles[e]+' R='+str(np.around(pcorr,2)), fontsize = 6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.scatter(Ytest[:,e], Ypred[:,e], c = 'slategrey', alpha = 0.7, s = 6)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        limx, limy = ax.get_xlim(), ax.get_ylim()
        lim = [max(limx[0], limy[0]), min(limx[1], limy[1])]
        ax.plot(lim, lim, color = 'dimgrey', ls = '-')
        if include_lr:
            lr = linear_model.LinearRegression().fit(Ytest[:, [e]], Ypred[:,e])
            ax.plot(np.array(limx), lr.predict(np.array(limx).reshape(-1,1)), color = 'r')
        if include_mainvar:
            centerx, centery = np.mean(Ytest[:,e]), np.mean(Ypred[:,e])
            maindir, u, v = np.linalg.svd(np.array([Ytest[:,e]-centerx, Ypred[:,e]-centery]), full_matrices=False)
            maindir = maindir[:,0]
            slope = maindir[1]/maindir[0]
            bias = centery-slope*centerx
            ax.plot(np.array(limx), np.array(limx)*slope + bias, color = 'r')
        
    if xlabel is not None:
        fig.text(0.5, 0.05-0.25/y_row, xlabel, ha='center')
    if ylabel is not None:
        fig.text(0.05-0.2/x_col, 0.5, ylabel, va='center', rotation='vertical')
    if outname is not None:
        print('SAVED as', outname)
        fig.savefig(outname, dpi = 200, bbox_inches = 'tight')
    else:
        return fig
