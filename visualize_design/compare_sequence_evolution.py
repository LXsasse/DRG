# compare_sequence_evolution.py
import numpy as np
import sys, os
import matplotlib.pyplot as plt

evo_files = sys.argv[1]
if ',' in evo_files:
    evo_files =evo_files.split(',')
else:
    evo_files =[evo_files]

nmethods = len(evo_files)

evo_ev = []
for e, evo in enumerate(evo_files):
    obj = open(evo, 'r').readlines()
    opt = []
    for l,line in enumerate(obj):
        line = line.strip().split(';')
        op = []
        for ent in line:
            if '--' in ent:
                ent = ent.split(',')[-1].split("-")
                op.append(['-'.join(ent[:-2]), '-'.join(ent[-2:])])
            else:
                op.append(ent.split(',')[-1].rsplit('-',1))
        opt.append(np.array(op, dtype = float))
    if len(opt) > 0:
        evo_ev.append(opt)
    else:
        print(evo, 'has no content')
        sys.exit()
        
nconditions = len(evo_ev[0][0][0])

name_methods = ['' for i in range(len(evo_files))]
if '--name_methods' in sys.argv:
    name_methods = sys.argv[sys.argv.index('--name_methods')+1].split(',')

minevi = 3

evo_med = []
for e, evo in enumerate(evo_ev):
    pos = [[] for c in range(nconditions)]
    for seq in evo:
        for p, pseq in enumerate(seq):
            for n in range(nconditions):
                if p == len(pos[n]):
                    pos[n].append([])
                pos[n][p].append(pseq[n])
    emed = []
    for n in range(nconditions):
        #print(pos[n])
        mean, var = [], []
        for seqs in pos[n]:
            if len(seqs) >= minevi:
                mean.append(np.mean(seqs))
                var.append(np.std(seqs))
        emed.append([np.array(mean),np.array(var)])
    evo_med.append(emed)



ylabel = None
if '--ylabels' in sys.argv:
    ylabel = sys.argv[sys.argv.index('--ylabels')+1].split(',')

targets = None
if '--targets' in sys.argv:
    targets = np.array(sys.argv[sys.argv.index('--targets')+1].split(','), dtype = float)
    
colors = ['olive', 'darksalmon','limegreen', 'purple', 'maroon', 'goldenrod', 'cornflowerblue',  'darkslategrey']

columns = int(np.sqrt(nconditions))
rows = int(nconditions/columns) + int(nconditions%columns > 0)
fig = plt.figure(figsize = (3.5*columns,3.5*rows), dpi = int(200/np.sqrt(rows)))
lim = []
axs = []
for n in range(nconditions):
    ax = fig.add_subplot(rows, columns, n+1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    lenlines = []
    linesstart = []
    for l, lines in enumerate(evo_med):
        lenlines.append(len(lines[n][0]))
        linesstart.append(lines[n][0][0])
        ax.scatter(np.arange(len(lines[n][0])), lines[n][0], c = colors[l], s = 2, label = name_methods[l])
        #print(lines[n][0], lines[n][1])
        ax.fill_between(np.arange(len(lines[n][0])), lines[n][0]+lines[n][1], lines[n][0] - lines[n][1], color = colors[l], alpha = 0.3)
    get_xlim = ax.get_xlim()
    ax.plot(get_xlim, [np.mean(linesstart), np.mean(linesstart)], color = 'grey', alpha = 0.5)
    if int(n/columns) == rows-1:
        ax.set_xlabel('Steps')
    if n%columns == 0:
        if ylabel is not None:
            ax.set_ylabel('Activity ('+ylabel[n]+')')
        else:
            ax.set_ylabel('Activity')
    if targets is not None:
        ax.plot(get_xlim, [targets[n], targets[n]], c = 'red', ls = '--')
    lim.append(ax.get_ylim())
    axs.append(ax)
    if n == 0 and '--name_methods' in sys.argv:
        ax.legend(markerscale = 2, prop={'size' : 6})
if '--scaley' in sys.argv:
    limtot = [np.amin(np.array(lim)[:,0]), np.amax(np.array(lim)[:,1])]
    for ax in axs:
        ax.set_ylim(limtot)

if '--savefig' in sys.argv:
    outname = sys.argv[sys.argv.index('--savefig')+1]
    fig.savefig(outname, dpi = 200, bbox_inches = 'tight')

fig.tight_layout()
plt.show()




