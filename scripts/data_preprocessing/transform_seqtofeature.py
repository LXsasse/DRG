'''
one hot encoder takes a txt file of sequences converts them into one-hot encoding, and saves it as npz file
can also generate k-mer profile instead
offers option to use numpy sparse matrices to deal with more sequences
also allows to annotate parts of the sequence with additional information such as genetic region (5', 3', exon, intron)
    # gentic regions are provided as Gene_name, region_name, location, total_length_of_gene
'''

import numpy as np
import sys, os
import time

from drg_tools.io_utils import readinfasta, readinlocation
from drg_tools.sequence_utils import make_kmer_representation as kmer_rep
from drg_tools.sequence_utils import seqlen, quick_onehot
    



if __name__ == '__main__':
    if '--mprocessing' in sys.argv:
        mprocessing = True
        num_cores = int(sys.argv[sys.argv.index('--mprocessing')+1])
    else:
        mprocessing = False
        num_cores = 1
        
    # input files
    fastafile = sys.argv[1]
    genenames, sequences = readinfasta(fastafile)

    outname = os.path.splitext(fastafile)[0]

    selen = seqlen(sequences)
    print('Max sequence length', genenames[np.argmax(selen)], np.amax(selen))
    
    if '--filter_genelength' in sys.argv:
        maxsize = int(sys.argv[sys.argv.index('--filter_genelength')+1])
        filt = np.array(selen) <= maxsize
        genenames, sequences = genenames[filt], sequences[filt]
        outname += '_max'+str(maxsize)
        print(int(np.sum(~filt)), 'removed because longer than', maxsize)
    
    elif '--cut_seqlength' in sys.argv:
        maxsize = int(sys.argv[sys.argv.index('--cut_seqlength')+1])
        filt = np.where(np.array(selen) > maxsize)[0]
        for s in filt:
            sequences[s] = sequences[s][:maxsize]
        outname += '_cut'+str(maxsize)
        print(len(filt), 'shortened because longer than', maxsize)

    # optionally provide positions with gneomic regions in transcripts
    # Format:
    # genomic regions are provided as Gene_name, region_name, location, total_length_of_gene
    #### !!! adapt to also provide locations for open RNA fragments, have several locations basically
    genreghot = None
    gregions = None
    if '--genomicregions' in sys.argv:
        regfile = sys.argv[sys.argv.index('--genomicregions')+1]
        gennames, genreghot, gregions = readinlocation(regfile)
        # sort to genenames
        seqsort = np.argsort(genenames)[np.isin(np.sort(genenames), gennames)]
        regsort = np.argsort(gennames)[np.isin(np.sort(gennames), genenames)]
        genenames, sequences = genenames[seqsort], sequences[seqsort]
        gennames, genreghot = gennames[regsort], genreghot[regsort]
        print(np.shape(genenames), np.shape(sequences), np.shape(genreghot))
        outname += '_genreg-'+os.path.splitext(os.path.split(regfile)[1])[0]


    # Split sequence into k-mers
    if '--kmers' in sys.argv:
        ftype = sys.argv[sys.argv.index('--kmers')+1] # regular, decreasing, gapped, mismatch
        klen = int(sys.argv[sys.argv.index('--kmers')+2])
        gaplen = 0
        if ftype in ['gapped', 'mismatch']:
            gaplen = int(sys.argv[sys.argv.index('--kmers')+3])
        
        # if other than ACGT
        nucleotides = 'ACGT'
        if '--nucleotides' in sys.argv:
            nucleotides = sys.argv[sys.argv.index('--nucleotides')+1]
        
        datatype = np.int8
        if '--highint' in sys.argv:
            datatype = int
        
        outname += '_kmer-'+ftype+str(klen)+'-'+str(gaplen)
        seqfeatures = kmer_rep(sequences, ftype, klen, gaplen, nucleotides = nucleotides, onehotregion = genreghot, datatype = datatype, mprocessing = mprocessing, num_cores =num_cores)
        
    else:
        # If sequences contain nucleotides that can represent all others
        wildcard_element = None
        if '--wildcard' in sys.argv:
            wildcard_element = sys.argv[sys.argv.index('--wildcard_element')+1]
        nucleotides = 'ACGT'
        # if other than ACGT
        if '--nucleotides' in sys.argv:
            nucleotides = sys.argv[sys.argv.index('--nucleotides')+1]
        
        outname += '_onehot-'+nucleotides
        
        alignto = 'left'
        if '--align_sequence' in sys.argv:
            alignto = sys.argv[sys.argv.index('--align_sequence')+1] # left, right, bidirectional
        outname += '_align'+alignto
        
        #t1 = time.time()
        #seqfeatures = onehot(sequences, nucleotides, wildcard = wildcard_element, onehotregion = genreghot)
        #t2 = time.time()
        seqfeatures = quick_onehot(sequences, nucleotides, wildcard = wildcard_element, onehotregion = genreghot, region_names = gregions, align = alignto)
    
    seqfeatures, featurenames = seqfeatures
    # save as npz file containing features and gene names
    print('Saved as \n'+outname+'.npz')
    np.savez_compressed(outname+'.npz', seqfeatures = seqfeatures, genenames = genenames, featurenames = featurenames)






