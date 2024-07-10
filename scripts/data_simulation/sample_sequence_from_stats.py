import numpy as np
import sys, os
import matplotlib.pyplot as plt 
import time

def read_probs(prob_file):
    stats = np.genfromtxt(prob_file, delimiter = '\t', dtype = str)
    segments = stats[:, 0]
    probs = stats[:, 1].astype(float)
    probs = probs/np.sum(probs)
    cond_probs = np.zeros((len(probs), 4))
    nucs = np.array(list('ACGT'))
    for s, seg in enumerate(segments):
        for c, seg1 in enumerate(segments):
            if seg[1:] == seg1[:2]:
                cond_probs[s, nucs == seg1[-1]] += probs[c]
    cond_probs = cond_probs/np.sum(cond_probs, axis = 1)[:,None]
    return segments, probs, cond_probs

def sample_sequences(seqmat, probs):
    # seqmat: matrix with individual lengths of each region
    # probs: triplets with segments, probs, cond_probs in the order of the segments
    nucs = list('ACGT')
    n_seqs = len(seqmat)
    allseqs = []
    for n, seqlen in enumerate(seqmat):
        seqs = []
        for l, ls in enumerate(seqlen):
            sampletype = probs[l][-1]
            
            lobj = len(probs[l][0][0])
            prob = probs[l][1]
            tri = probs[l][0]
            cprob = probs[l][2]
            
            if sampletype:
                seqs.append(''.join(np.random.choice(tri, p = prob, size = int(ls/lobj))))
            else:
                seq = np.random.choice(tri, p = prob)
                steps = ls - lobj
                for s in range(steps):
                    si = list(tri).index(seq[-lobj:])
                    seq += np.random.choice(nucs, p = cprob[si])
                seqs.append(seq)
        allseqs.append(''.join(np.array(seqs)))        
    return allseqs


def add_sequencemarks(allseqs, to_add, loc):
    
    if isinstance(to_add, str):
        to_add = [to_add for t in range(len(allseqs))]
    elif len(to_add) != len(allseqs):
        to_add = np.random.choice(to_add, len(allseqs))
    
    ladd = len(to_add[0])
    
    if isinstance(loc, int):
        loc = np.ones(len(allseqs), dtype = int)*loc
    elif len(loc) != len(allseqs):
        loc = np.random.choice(loc, len(allseqs))
    
    for s, seqs in enumerate(allseqs):
        #if s == 0:
            #print(loc[s], to_add[s], ladd)
        allseqs[s] = seqs[:loc[s]]+ to_add[s] +seqs[loc[s]+ladd:]
      
    return allseqs



if '--realistic_sequences' in sys.argv:
    # load realistic triplet frequencies and sample from them.
    random_triplets, rprob, rconprob = read_probs(sys.argv[sys.argv.index('--realistic_sequences')+1])
    promoter_triplets, pprob, pconprob = read_probs(sys.argv[sys.argv.index('--realistic_sequences')+2])
    enhancer_triplets, eprob, econprob = read_probs(sys.argv[sys.argv.index('--realistic_sequences')+3])
    codon_triplets, cprob, cconprob = read_probs(sys.argv[sys.argv.index('--realistic_sequences')+4])
    utr3_triplets, u3prob, u3conprob = read_probs(sys.argv[sys.argv.index('--realistic_sequences')+5])
    utr5_triplets, u5prob, u5conprob = read_probs(sys.argv[sys.argv.index('--realistic_sequences')+6])
    n_seqs = 100
    l_enhancer = 100 # lengths of enhancers
    l_promoter = 100 # length of promoters
    l_5utr = 100 # length of 5utrs
    l_cds = 500 # length of cds
    l_3utr = 200 # length of 3utr
    l_insulator = 200 # length of dna elements that have no function
    l_seq = l_enhancer + l_insulator + l_promoter + l_5utr + l_cds + l_3utr + l_insulator + l_enhancer
    
    seqmat = np.zeros((n_seqs, 8), dtype = int)
    seqmat[:,0] = l_enhancer
    seqmat[:,1] = l_insulator - np.random.randint(0,l_insulator, n_seqs)
    seqmat[:,2] = l_promoter
    seqmat[:,3] = l_5utr
    seqmat[:,4] = l_cds - np.random.randint(l_cds/2,l_cds, n_seqs)
    seqmat[:,4] = seqmat[:,4] - seqmat[:,4]%3
    seqmat[:,5] = l_3utr
    seqmat[:,6] = l_insulator - np.random.randint(0, l_insulator, n_seqs)
    seqmat[:,7] = l_enhancer
    
    tsss = np.sum(seqmat[:,:3], axis = 1)
    stcdn = np.sum(seqmat[:,:4], axis = 1)
    stpcdn = np.sum(seqmat[:,:5], axis = 1)
    clste = np.sum(seqmat[:,:6], axis = 1)
    
    probs = [[enhancer_triplets, eprob, econprob, False], [random_triplets, rprob, rconprob, False], [promoter_triplets, pprob, pconprob, False], [utr5_triplets, u5prob, u5conprob, False],[codon_triplets, cprob, cconprob, True], [utr3_triplets, u3prob, u3conprob, False], [random_triplets, rprob, rconprob, False], [enhancer_triplets, eprob, econprob, False]]
    
    t0 = time.time()
    sequences = sample_sequences(seqmat, probs)
    t1 = time.time()
    print(t1-t0)
    
    for s, seq in enumerate(sequences):
        sequences[s] = seq.lower()
    
    sequences = add_sequencemarks(sequences, ['GGGCGCC', 'GGCCGCC', 'GCCCGCC', 'CGGCGCC', 'GCGCGCC', 'CGCCGCC', 'CCGCGCC', 'CCCCGCC'], tsss- np.random.choice(np.arange(35,55, dtype = int), n_seqs))
    sequences = add_sequencemarks(sequences, 'TATAA', tsss - np.random.choice(np.arange(25, 35, dtype = int), n_seqs))
    sequences = add_sequencemarks(sequences, ['CCACC','CCGCC', 'GCACC', 'CCAAC', 'CCGAC'], stcdn-5)
    t2 = time.time()
    print(t2-t1)
    sequences = add_sequencemarks(sequences, 'AUG', stcdn)
    sequences = add_sequencemarks(sequences, ['UAG', 'UAA', 'UGA'],stpcdn)
    sequences = add_sequencemarks(sequences, ['AATAAA', 'ATTAAA'], clste -np.random.choice(np.arange(20,50, dtype =int), n_seqs))
    
    
    for i in range(int(len(sequences[0])/50)):
        print(sequences[0][i*50:(i+1)*50], 50*(i+1))
    print(stcdn[0], tsss[0], stpcdn[0], clste[0])
    t3 = time.time()
    print(t3-t2)
    



''''
nucs = np.array(list('ACGT'))
stats = np.genfromtxt(sys.argv[1], delimiter = '\t', dtype = str)
segments = stats[:, 0]
probs = stats[:, 1].astype(float)

cond_probs = np.zeros((len(probs), 4))

for s, seg in enumerate(segments):
    for c, seg1 in enumerate(segments):
        if seg[1:] == seg1[:2]:
            cond_probs[s, nucs == seg1[-1]] += probs[c]
            
cond_probs = cond_probs/np.sum(cond_probs, axis = 1)[:,None]

n_seq = 3000
l_seq = 200
seqs = []
seqcounts = np.zeros(len(segments))
for n in range(n_seq):
    if n %100 == 0:
        print(n)
    seq = np.random.choice(segments, p = probs)
    for s in range(l_seq - 3):
        si = list(segments).index(seq[-3:])
        seq += np.random.choice(nucs, p = cond_probs[si])
        seqcounts[si] += 1
    seqs.append(seq)

seqcounts = seqcounts/np.sum(seqcounts)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(np.arange(len(segments)), probs, alpha = 0.4)
ax.bar(np.arange(len(segments)), seqcounts, alpha = 0.4)
ax.set_xticks(np.arange(len(segments)))
ax.set_xticklabels(segments, rotation = 90)
plt.show()
'''
