# one hot encoder takes a txt file of sequences converts them into one-hot encoding, and saves it as npz file
# can also generate k-mer profile instead
# offers option to use numpy sparse matrices to deal with more gene sequences
# also allows to annotate parts of the sequence with additional information such as genetic region (5', 3', exon, intron)
    # gentic regions are provided as Gene_name, region_name, location, total_length_of_gene

import numpy as np
import sys, os
import time
from joblib import Parallel, delayed
import multiprocessing

if '--mprocessing' in sys.argv:
    mprocessing = True
    num_cores = int(sys.argv[sys.argv.index('--mprocessing')+1])
else:
    mprocessing = False
    num_cores = 1




# reads in fasta file
def readinfasta(fatafile, minlen = 10):
    obj = open(fastafile, 'r').readlines()
    genes = []
    sequences = []
    for l, line in enumerate(obj):
        if line[0] == '>':
            sequence = obj[l+1].strip()
            if sequence != 'Sequence unavailable' and len(sequence) > minlen:
                genes.append(line[1:].strip())
                sequences.append(sequence.upper())
    sortgen = np.argsort(genes)
    genes, sequences = np.array(genes)[sortgen], np.array(sequences)[sortgen]
    return genes, sequences


    
# reads in genetic location file: format is as follows:
#     Gene_name, region_name, location, total_length_of_gene
# adds zeros to end of encoding for shorter sequences
def readinlocation(regfile):
    obj = np.genfromtxt(regfile, dtype = str, delimiter = '\t')
    possible_regions = list(np.unique(obj[:,1]))
    genenames = obj[:,0]
    gsort = np.argsort(genenames)
    genes = list(np.unique(genenames))
    sequences = np.zeros((len(genes), np.amax(obj[:,-1].astype(int)), len(possible_regions)), dtype = np.int8)
    for l, line in enumerate(obj[gsort]):
        if ',' in line[2]:
            sequences[genes.index(line[0]), np.array(line[2].split(','), dtype = int), possible_regions.index(line[1])] =1
        else:
            sequences[genes.index(line[0]), int(line[2].split('-')[0]):int(line[2].split('-')[1]), possible_regions.index(line[1])] =1
    return np.array(genes), sequences, np.array(possible_regions)
    


def seqlen(arrayofseqs):
    return np.array([len(seq) for seq in arrayofseqs])

def check_addonehot(onehotregion, shapeohvec1, selen):
    # check conditions if one-hot encoded regions can be added
    addonehot = False
    if onehotregion is not None:
        if np.shape(onehotregion)[1] == shapeohvec1 :
            addonehot = True
        else:
            print("number of genetic regions do not match sequences")
            sys.exit()
    return addonehot


# generates one-hot encoding by comparing arrays
def quick_onehot(sequences, nucs = 'ACGT', wildcard = None, onehotregion = None, region_names = None, align = 'left'):
    selen = seqlen(sequences)
    nucs = np.array(list(nucs))
    if align == 'bidirectional':
        mlenseqs = 2*np.amax(selen) + 20
    else:
        mlenseqs = np.amax(selen) 
    ohvec = np.zeros((len(sequences),mlenseqs, len(nucs)), dtype = np.int8)
    
    if wildcard is not None:
        nucs.append(wildcard)
    nucs = np.array(nucs)
    # check conditions if one-hot encoded regions can be added
    addonehot = check_addonehot(onehotregion, mlenseqs, selen)

    if align == 'left' or align == 'bidirectional':
        for s, sequence in enumerate(sequences):
            ohvec[s][:len(sequence)] = np.array(list(sequence))[:, None] == nucs
    if align == 'right' or align == 'bidirectional':
        for s, sequence in enumerate(sequences):
            ohvec[s][-len(sequence):] = np.array(list(sequence))[:, None] == nucs
    ohvec = ohvec.astype(np.int8)
    
    if addonehot:
        ohvec = np.append(ohvec, onehotregion, axis = -1)
        nucs = np.append(nucs, region_names)
        print(nucs)
    return ohvec, nucs

# generates one-hot encoding with for loop
def onehot(sequences, nucs = 'ACGT', wildcard = None, onehotregion = None, align = 'left'):
    selen = seqlen(sequences)
    nucs = list(nucs)
    if align == 'bidirectional':
        mlenseqs = 2*np.amax(selen) + 20
    else:
        mlenseqs = np.amax(selen) 
    ohvec = np.zeros((len(sequences),mlenseqs, len(nucs)), dtype = np.int8)
    nucsvec = np.eye(len(nucs), dtype = np.int8)
    
    if wildcard is not None:
        nucs.append(wildcard)
        nucsvec = np.append(nucsvec, [np.ones(len(nucs), dtype =np.int8)], axis = 0)

    # check conditions if one-hot encoded regions can be added
    addonehot = check_addonehot(onehotregion, mlenseqs, selen)

    if align == 'left' or align == 'bidirectional':
        for s, sequence in enumerate(sequences):
            for n, nt in enumerate(sequence):
                ohvec[s, n] = nucsvec[nucs.index(nt)]
    if align == 'right' or align == 'bidirectional':
        for s, sequence in enumerate(sequences):
            for n, nt in enumerate(sequence):
                ohvec[s, n-len(sequence)] = nucsvec[nucs.index(nt)]
    
    if addonehot:
        ohvec = np.append(ohvec, onehotregion, axis = -1)
    return ohvec
    
def add_sing(arr, sing):
    outarr = []
    for ar in arr:
        for si in sing:
            outarr.append(ar+si)
    return outarr

def def_region(onehotregkmer):
    region, regionnum = np.unique(np.nonzero(onehotregkmer)[-1], return_counts = True)
    region = region[np.argmax(regionnum)]
    return region
    


def kmer_rep(sequences, kmertype, kmerlength, gapsize, nucleotides = 'ACGT', onehotregion = None, datatype = np.int8):
    kmers = list(nucleotides)
    
    # check conditions if one-hot encoded regions can be added
    selen = seqlen(sequences)
    mlenseqs = np.amax(selen) 
    addonehot = check_addonehot(onehotregion, mlenseqs, selen)
    ohvec = 0
    
    # regular kmer counts of lenght kmerlength
    if kmertype == 'regular':
        for i in range(kmerlength-1):
            kmers = add_sing(kmers, list(nucleotides))
        
        
        features = np.zeros((len(sequences), len(kmers)), dtype = datatype)
        if addonehot:
            features = np.zeros((len(sequences), len(kmers))*np.shape(onehotregion)[-1], dtype = datatype)
            kmers = np.concatenate([[kmer+'_'+greg for kmer in kmers] for greg in gregions])
        kmers = np.sort(kmers)
        if mprocessing:
            if not addonehot:
                def findkmer(seq, kmerlength,kmers, features, s):
                    featlist = []
                    for k in range(len(seq)-kmerlength+1):
                        featlist.append(seq[k:k+kmerlength])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                    return features, s
            else:
                def findkmer(seq, kmerlength,kmers, features, s):
                    for k in range(len(seq)-kmerlength+1):
                        region = def_region(onehotregion[s, k+kmerlength])
                        featlist.append(seq[k:k+kmerlength]+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                    return features, s
            
            results = Parallel(n_jobs=num_cores)(delayed(findkmer)(sequences[i], kmerlength, kmers, features[i], i) for i in range(len(sequences)))
            
            for rvec,r in results:
                features[r] = rvec
            
        else:
            for s, seq in enumerate(sequences):
                featlist = []
                if not addonehot:
                    for k in range(len(seq)-kmerlength+1):
                        featlist.append(seq[k:k+kmerlength])
                    featlist, featcount = np.unique(featlist, return_counts = True)
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
            
                else:
                    for k in range(len(seq)-kmerlength+1):
                        region = def_region(onehotregion[s, k+kmerlength])
                        rfeatlist.append(seq[k:k+kmerlength]+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                
    # all kmers of length 2 to kmerlength
    elif kmertype == 'decreasing':
        kmerlist = []
        for i in range(kmerlength-1):
            kmers = add_sing(kmers, list(nucleotides))
            kmerlist.append(kmers)
        
        kmers = list(np.concatenate(kmerlist))
        features = np.zeros((len(sequences), len(kmers)), dtype = datatype)
        if addonehot:
            features = np.zeros((len(sequences), len(kmers))*np.shape(onehotregion)[-1], dtype = datatype)
            kmers = np.concatenate([[kmer+'_'+greg for kmer in kmers] for greg in gregions])
        kmers = np.sort(kmers)
        if mprocessing:
            if not addonehot:
                def findkmer(seq, kmerlength, kmers,features, s):
                    featlist = []
                    
                    for kl in range(2, kmerlength+1):
                        for k in range(len(seq)-kl+1):
                            featlist.append(seq[k:k+kl])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]   
                    return features, s
            else:
                def findkmer(seq, kmerlength, kmers, features, s):
                    featlist = []
                    for kl in range(2, kmerlength+1):
                        for k in range(len(seq)-kl+1):
                            region = def_region(onehotregion[s, k:k+kl])
                            featlist.append(seq[k:k+kl]+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                    return features, s
            
            results = Parallel(n_jobs=num_cores)(delayed(findkmer)(sequences[i], kmerlength, kmers, features[i], i) for i in range(len(sequences)))
            
            for rvec, r in results:
                features[r] = rvec
        else:
            for s, seq in enumerate(sequences):
                featlist = []
                if addonehot:
                    for kl in range(2, kmerlength+1):
                        for k in range(len(seq)-kl+1):
                            region = def_region(onehotregion[s, k:k+kl])
                            featlist.append(seq[k:k+kl]+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                else:
                    for kl in range(2, kmerlength+1):
                        for k in range(len(seq)-kl+1):
                            featlist.append(seq[k:k+kl])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
        
    elif kmertype == 'gapped':
        
        for i in range(kmerlength-1-gapsize):
            kmers = add_sing(kmers, list(nucleotides))
        features = np.zeros((len(sequences), len(kmers)), dtype = datatype)
        if addonehot:
            features = np.zeros((len(sequences), len(kmers))*np.shape(onehotregion)[-1], dtype = datatype)
            kmers = np.concatenate([[kmer+'_'+greg for kmer in kmers] for greg in gregions])
        kmers = np.sort(kmers)
        
        if mprocessing:
            if not addonehot:
                def findkmer(seq, kmerlength, kmers,features, s):
                    featlist = []
                    for k in range(len(seq)-kmerlength+1):
                        for g in range(1, kmerlength-gapsize-1):
                            featlist.append(seq[k:k+g] + seq[k+g+gapsize:k+kmerlength])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]        
                    return features, s
            else:
                def findkmer(seq, kmerlength, kmers, features, s):
                    featlist= []
                    for k in range(len(seq)-kmerlength+1):
                        for g in range(1, kmerlength-gapsize-1):
                            region = def_region(np.append(onehotregion[s, k:k+g], onehotregion[s, k+g+gapsize:k+kmerlength], axis = 0))
                            featlist.append(seq[k:k+g] + seq[k+g+gapsize:k+kmerlength]+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                    return features, s
            
            results = Parallel(n_jobs=num_cores)(delayed(findkmer)(sequences[i], kmerlength, kmers, features[i], i) for i in range(len(sequences)))
            
            for rvec, r in results:
                features[r] = rvec
        
        else:
            for s, seq in enumerate(sequences):
                featlist = []
                if addonehot:
                    for k in range(len(seq)-kmerlength+1):
                        for g in range(1, kmerlength-gapsize-1):
                            region = def_region(np.append(onehotregion[s, k:k+g], onehotregion[s, k+g+gapsize:k+kmerlength], axis = 0))
                            featlist.append(seq[k:k+g] + seq[k+g+gapsize:k+kmerlength]+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                else:
                    for k in range(len(seq)-kmerlength+1):
                        for g in range(1, kmerlength-gapsize-1):
                            featlist.append(seq[k:k+g] + seq[k+g+gapsize:k+kmerlength])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                
    elif kmertype == 'mismatch':
        # generate kmerlist
        print( 'Generate kmer')
        
        for i in range(kmerlength-1-gapsize):
            kmers = add_sing(kmers, list(nucleotides))
        rkmerfeat = np.copy(kmers)
        kmers = []
        for g in range(gapsize):
            gappedkmerfeat = []
            for kmer in rkmerfeat:
                gkmer = []
                for i in range(1,len(kmer)):
                    gkmer.append(kmer[:i]+'X'+kmer[i:])
                gappedkmerfeat.append(gkmer)
            rkmerfeat = np.unique(np.concatenate(gappedkmerfeat))
        kmers = list(rkmerfeat)
        print( 'kmers done ...')
    
        # generate masks for wildcard elements:
        mask = np.zeros(kmerlength) > 0.
        masks = [mask]
        for g in range(gapsize):
            gapmasks = []
            for mask in masks:
                for i in range(1,len(mask)-1):
                    if mask[i] == False:
                        comask = np.copy(mask)
                        comask[i] = True
                        gapmasks.append(comask)
            masks = list(np.unique(gapmasks, axis = 0))
        
        features = np.zeros((len(sequences), len(kmers)), dtype = datatype)
        if addonehot:
            features = np.zeros((len(sequences), len(kmers))*np.shape(onehotregion)[-1], dtype = datatype)
            kmers = np.concatenate([[kmer+'_'+greg for kmer in kmers] for greg in gregions])
        kmers = np.sort(kmers)
        if mprocessing:
            if not addonehot:
                def findkmer(seq, kmerlength, kmers,features, s):
                    featlist = []
                    for k in range(len(seq)-kmerlength+1):
                        for mask in masks:
                            mkmer = np.copy(seq[k:k+kmerlength])
                            mkmer[mask] = 'X'
                            featlist.append(''.join(mkmer))
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[ np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]  
                    return features, s
            else:
                def findkmer(seq, kmerlength, kmers, features, s):
                    featlist = []
                    for k in range(len(seq)-kmerlength+1):
                        region = def_region(onehotregion[s, k+kmerlength])
                        for mask in masks:
                            mkmer = np.copy(seq[k:k+kmerlength])
                            mkmer[mask] = 'X'
                            featlist.append(''.join(mkmer)+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[ np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                    return features, s
            
            results = Parallel(n_jobs=num_cores)(delayed(findkmer)(np.array(list(sequences[i])), kmerlength, kmers, features[i], i) for i in range(len(sequences)))
            
            for rvec, r in results:
                features[r] = rvec
        
        else:
            for s, seq in enumerate(sequences):
                featlist = []
                seq = np.array(list(seq))
                if addonehot:
                    for k in range(len(seq)-kmerlength+1):
                        region = def_region(onehotregion[s, k+kmerlength])
                        for mask in masks:
                            mkmer = np.copy(seq[k:k+kmerlength])
                            mkmer[mask] = 'X'
                            featlist.append(''.join(mkmer)+'_'+gregions[region])
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
                else:
                    for k in range(len(seq)-kmerlength+1):
                        for mask in masks:
                            mkmer = np.copy(seq[k:k+kmerlength])
                            mkmer[mask] = 'X'
                            featlist.append(''.join(mkmer))
                    featlist, featcount = np.unique(featlist, return_counts = True)    
                    features[s, np.isin(kmers, featlist)] = featcount[np.isin(featlist, kmers)]
        
    return features, kmers
    
# input files
fastafile = sys.argv[1]
genenames, sequences = readinfasta(fastafile)

outname = os.path.splitext(fastafile)[0]

if '--filter_genelength' in sys.argv:
    maxsize = int(sys.argv[sys.argv.index('--filter_genelength')+1])
    selen = seqlen(sequences)
    filt = np.array(selen) <= maxsize
    genenames, sequences = genenames[filt], sequences[filt]
    outname += '_max'+str(maxsize)
    print(int(np.sum(~filt)), 'removed because longer than', maxsize)
    

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
    seqfeatures = kmer_rep(sequences, ftype, klen, gaplen, nucleotides = nucleotides, onehotregion = genreghot, datatype = datatype)
    
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
    
# save as npz file containing features and gene names
print('Saved as \n'+outname+'.npz')
np.savez_compressed(outname+'.npz', seqfeatures = seqfeatures, genenames = genenames)






