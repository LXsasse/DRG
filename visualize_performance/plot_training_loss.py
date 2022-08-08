import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib import cm

valfiles=sys.argv[1].split(',')
modnames=sys.argv[2].split(',')


losses = []
for v, valf in enumerate(valfiles):
    loss = np.genfromtxt(valf, dtype = float, delimiter = '\t')
    nanmask = np.unique(np.where(~np.isnan(loss))[0])
    losses.append(loss[nanmask])
    if '--percentages' in sys.argv:
        losses[-1][:,0] = losses[-1][:,0]/np.amax(losses[-1][:,0])
    
fig = plt.figure(figsize = (9,9), dpi = 180)
cmap = np.append(cm.tab20b(np.arange(0,20,4)),cm.tab20c(np.arange(0,20,4)),axis = 0)

ax0 = fig.add_subplot(10, 1, 1)
if '--combine_sets' in sys.argv:
    width = 0.4
else:
    width = 0.8
ax0.set_position([0.1, 0.81, width, 0.14])
ax0.tick_params(left = False, labelleft = False, labelbottom = False, bottom = False)
rows = min(len(modnames),4)
columns = int(len(modnames)/rows)+int(len(modnames)%rows>0)
ax0.set_xlim([-0.1, columns])
ax0.set_ylim([-0.5, rows-0.5])
for m, name in enumerate(modnames):
    ax0.scatter([int(m/rows)], [rows - 1 -m%rows], color = cmap[m])
    ax0.text(int(m/rows)+0.02*columns, rows - 1 -m%rows, name ,va = 'center', ha = 'left', fontsize = 8)


    
ax1 = fig.add_subplot(221)
ax1.set_position([0.1, 0.45, 0.38, 0.32])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.tick_params(labelbottom = False)
for l, loss in enumerate(losses):
    ax1.plot(loss[:,0], loss[:,1], marker = '.', ms = 5, alpha = 0.5, color = cmap[l])
if '--combine_sets' in sys.argv:
    ax1.set_ylabel('Training Loss')
else:
    ax1.set_ylabel('Training Loss (val)')
if '--logx' in sys.argv:
    ax1.set_xscale('symlog')
ax1.grid()


ax2 = fig.add_subplot(223)
ax2.set_position([0.1, 0.1, 0.38, 0.32])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
for l, loss in enumerate(losses):
    ax2.plot(loss[:,0], loss[:,2], marker = '.', alpha = 0.5, ms = 5, color = cmap[l])
if '--combine_sets' in sys.argv:
    ax2.set_ylabel('Validation Loss')
else:
    ax2.set_ylabel('Validation Loss (val)')
if '--percentages' in sys.argv:
    ax2.set_xlabel('% Total Epochs')
else:
    ax2.set_xlabel('Epochs')
ax2.grid()
if '--logx' in sys.argv:
    ax2.set_xscale('symlog')

if '--combine_sets' in sys.argv:
    ax3 = ax1
    marker = '2'
else:
    ax3 = fig.add_subplot(222)
    ax3.set_position([0.52, 0.45, 0.38, 0.32])
    ax3.spines['left'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.tick_params(left = False, labelleft = False, right = True, labelright = True, labelbottom = False)
    marker = '.'
for l, loss in enumerate(losses):
    ax3.plot(loss[:,0], loss[:,3], marker = marker, alpha = 0.5, ms = 5, color = cmap[l], label = 'Train')

if '--combine_sets' not in sys.argv:
    ax3.set_ylabel('Traning Loss (train)')
    ax3.grid()
    if '--logx' in sys.argv:
        ax3.set_xscale('symlog')
else:
    ax3.legend()

if '--combine_sets' in sys.argv:
    ax4 = ax2
else:
    ax4= fig.add_subplot(224)
    ax4.set_position([0.52, 0.1, 0.38, 0.32])
    ax4.spines['left'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.tick_params(left = False, labelleft = False, right = True, labelright = True)

for l, loss in enumerate(losses):
    ax4.plot(loss[:,0], loss[:,4], marker = marker, alpha = 0.5, ms = 5, color = cmap[l],label = 'Train')

if '--combine_sets' not in sys.argv:
    if '--percentages' in sys.argv:
        ax4.set_xlabel('% Total Epochs')
    else:
        ax4.set_xlabel('Epochs')

    ax4.set_ylabel('Validation Loss (train)')
    ax4.grid()
    if '--logx' in sys.argv:
        ax4.set_xscale('symlog')
else:
    ax4.legend()
    
if '--adjust_axis' in sys.argv:
    ax1.set_ylim([min(ax1.get_ylim()[0],ax3.get_ylim()[0]), max(ax1.get_ylim()[1],ax3.get_ylim()[1])])
    ax2.set_ylim([min(ax2.get_ylim()[0],ax4.get_ylim()[0]), max(ax2.get_ylim()[1],ax4.get_ylim()[1])])
    if not '--combine_sets' in sys.argv:
        ax3.set_ylim([min(ax1.get_ylim()[0],ax3.get_ylim()[0]), max(ax1.get_ylim()[1],ax3.get_ylim()[1])])
        ax4.set_ylim([min(ax2.get_ylim()[0],ax4.get_ylim()[0]), max(ax2.get_ylim()[1],ax4.get_ylim()[1])])

if '--savefig' in sys.argv:
    fig.savefig(sys.argv[sys.argv.index('--savefig')+1], dpi = 200, bbox_inches = 'tight')
else:
    plt.show()
