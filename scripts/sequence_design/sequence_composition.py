import numpy as np
import sys, os
from seqtofeature_beta import kmer_rep
import matplotlib.pyplot as plt

def add_sing(arr, sing):
    outarr = []
    for ar in arr:
        for si in sing:
            outarr.append(ar+si)
    return outarr

def kmerseq(n):
    kmers = list('ACGT')
    for i in range(kmerlength-1):
        kmers = add_sing(kmers, list('ACGT'))
    kmers = np.sort(kmers)
    return kmers


def plot_dist(kmers, kmerfrac, kmerfrac2 = None):
    fig = plt.figure(figsize = (len(kmers)*0.3, 2.5), dpi = 200)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if kmerfrac2 is not None:
        ax.bar(np.arange(len(kmers)), kmerfrac, alpha = 0.45)
        ax.bar(np.arange(len(kmers)), kmerfrac2, alpha = 0.45)
    else:
        ax.bar(np.arange(len(kmers)), kmerfrac)
    ax.set_xticks(np.arange(len(kmers)))
    ax.set_xticklabels(kmers, rotation = 90)
    ax.set_ylabel('Fraction')
    return fig

def write(kmers, kmerfrac, name):
    obj = open(name, 'w')
    for k, kmer in enumerate(kmers):
        obj.write(kmer+'\t'+str(round(kmerfrac[k],4))+'\n')
    obj.close()

if '--compare' in sys.argv:
    a = np.genfromtxt(sys.argv[1], dtype = str, delimiter = '\t')
    b = np.genfromtxt(sys.argv[2], dtype = str, delimiter = '\t')
    outname = os.path.splitext(sys.argv[1])[0] + 'vs'+os.path.splitext(os.path.split(sys.argv[2])[1])[0]
    a, b = a[np.argsort(a[:,0])], b[np.argsort(b[:,0])]
    print(outname, a[:,0], b[:,0], len(a), len(b))
    if np.array_equal(a[:,0], b[:,0]):
        kmers = a[:,0]
        kfrac = a[:,1].astype(float)
        kfrac2 = b[:,1].astype(float)
        fig = plot_dist(kmers, kfrac, kmerfrac2 = kfrac2)
        fig.savefig(outname+'.jpg', dpi = 200, bbox_inches = 'tight')
        
    
else:
    fasta = open(sys.argv[1], 'r').readlines()
    if len(sys.argv) > 3:
        kmertype = sys.argv[2]
        kmerlength = int(sys.argv[3])
    else:
        kmertype = 'ALL'
            
    outname = os.path.splitext(sys.argv[1])[0]

    # readin fasta
    fname = []
    fseq = []
    for l, line in enumerate(fasta):
        if line[0] == '>':
            fname.append(line[1:].strip())
            fseq.append(fasta[l+1].strip())

    if kmertype == 'ALL':
        kmer2, nucs2 = kmer_rep(fseq, 'regular', 2, mprocessing = True, num_cores = 4)
        kmer2g, nucs2g = kmer_rep(fseq, 'mismatch', 3, gapsize = 1, mprocessing = True, num_cores = 4)
        kmer3, nucs3 = kmer_rep(fseq, 'regular', 3, mprocessing = True, num_cores = 4)
        kmer3g, nucs3g = kmer_rep(fseq, 'mismatch', 4, gapsize = 1, mprocessing = True, num_cores = 4)
        kmer2g2, nucs2g2 = kmer_rep(fseq, 'mismatch', 4, gapsize = 2, mprocessing = True, num_cores = 4)

        kmer2 = np.sum(kmer2, axis = 0)/np.sum(kmer2)
        kmer2g = np.sum(kmer2g, axis = 0)/np.sum(kmer2g)
        kmer3 = np.sum(kmer3, axis = 0)/np.sum(kmer3)
        kmer2g2 = np.sum(kmer2g2, axis = 0)/np.sum(kmer2g2)
        kmer3g = np.sum(kmer3g, axis = 0)/np.sum(kmer3g)

        fig2 = plot_dist(nucs2, kmer2)
        fig2g = plot_dist(nucs2g, kmer2g)
        fig3 = plot_dist(nucs3, kmer3)
        fig2g2 = plot_dist(nucs2g2, kmer2g2)
        fig3g= plot_dist(nucs3g, kmer3g)

        fig2.savefig(outname+'_dicomp.jpg', bbox_inches = 'tight', dpi = 200)
        fig2g.savefig(outname+'_digapcomp.jpg', bbox_inches = 'tight', dpi = 200)
        fig3.savefig(outname+'_tricomp.jpg', bbox_inches = 'tight', dpi = 200)
        fig2g2.savefig(outname+'_digap2comp.jpg', bbox_inches = 'tight', dpi = 200)
        fig3g.savefig(outname+'_trigapcomp.jpg', bbox_inches = 'tight', dpi = 200)

        write(nucs2, kmer2, outname+'dicomp.txt')
        write(nucs2g, kmer2g, outname+'digapcomp.txt')
        write(nucs3, kmer3, outname+'tricomp.txt')
        write(nucs2g2, kmer2g2, outname+'digap2comp.txt')
        write(nucs3g, kmer3g, outname+'trigapcomp.txt')
        plt.show()

    elif kmertype == 'codon':
        nucs = kmerseq(3)
        nucs = np.sort(nucs)
        kmermat = np.zeros(len(nucs))
        kmerlist = []
        for s, seq in enumerate(fseq):
            for i in range(int(len(seq)/3)):
                kmerlist.append(seq[i*3:(i+1)*3])
        unk, unkn = np.unique(kmerlist, return_counts = True)
        print(unk, unkn)
        kmermat[np.isin(nucs,unk)] = unkn[np.isin(unk, nucs)]
        kmer = kmermat/np.sum(kmermat)
        fig = plot_dist(nucs, kmer)
        fig.savefig(outname+kmertype+'.jpg', bbox_inches = 'tight', dpi = 200)
        write(nucs, kmer, outname+kmertype+'.txt')
        plt.show()

    else:
        kmer, nucs = kmer_rep(fseq, kmertype, kmerlength, mprocessing = True, num_cores = 4)
        kmer = np.sum(kmer, axis = 0)/np.sum(kmer)
        fig = plot_dist(nucs, kmer)
        fig.savefig(outname+kmertype+str(kmerlength)+'.jpg', bbox_inches = 'tight', dpi = 200)
        write(nucs, kmer, outname+kmertype+str(kmerlength)+'.txt')
        plt.show()

                          
                    
