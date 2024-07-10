import numpy as np
import sys, os
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import time
from functions import correlation
from seqtofeature_beta import kmer_rep


# Realistic simulation of gene expression with enhancer, promoter, random, utr, cds regions
    # DNA and RNA motif dependent gene expression modeling
    # Transcription devidid by degradation rate as expression
    # Activators and repressors
    # Distance dependent enhancer activator_interactions
    # Variable length of random elements
    # Linear effects for rbps and tfs
    # Non-linear effects of rbps and tfs possible for subset

# Missing:
    # multiple enhancers/far distance enhancers
    # Enhancers that determine expression for several genes
    # introns/intronic-enhancers
    # 5'mrna effect, cds mrna effect
    # cell type specific splicing rate effect
    # cell type specific splicing effect on alternative regulatory elements
    # Biophysical binding equation for protein-D/RNA and protein-protein interactions
        # To include the protein and RNA levels in each cell type that are influenced by the active network.
    # Evolutionary sequence pattern, i.e. high concentration of motifs in enhancers and avoidance of motifs in other regions:
    # Evolutionary classes of Enhancers and promoters with different distributions of motifs
    # set of co-factors and the thee co-factor models that are described in Alex Stark paper. Co-factors only have protein interactions and need to be expressed. They also have a stronger effect if multiple preferred tfs are in the enhancer

# Extensions: 
    # Multiple individuals 
    # Multiple species
    # Mpra assay

nuctides = np.array(list('ACGT'))

np.random.seed(101)
# general structure of the sequence will be
# enhancer - insulator_variable_length - promoter - 5utr - cds -3utr - insulator_variable_length - enhancer
# motifs will be placed in enhancer, promoter, and 5utr regions

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

def compute_expression(promoter, list_enhancer, enhancer_dist, stability, exp_func = False, multiply_enhancer = False):
    random_degrad = 1. # degradation rate if no motif is placed in the transcript
    distance_var = 300
    min_scaling = 0.2
    enhancer_scaling = np.exp(-(enhancer_dist/distance_var)**2)
    enhancer_scaling[enhancer_scaling < min_scaling] = min_scaling
    if multiply_enhancer:
        KT = np.sum(promoter[:, None] * (list_enhancer * enhancer_scaling), axis = -1)
    else:
        KT = promoter + np.sum(list_enhancer * enhancer_scaling, axis = -1)
    KT[KT < 0] = 0
    KD = random_degrad + stability
    KD[KD < 1] = 1
    if exp_func:
        expression = np.exp(KT-KD)
    else:
        expression = KT/KD
    return expression, KT, KD


def kmer_to_pwm(kmerlist):
    pwm = np.zeros((len(kmerlist[0]), 4))
    for kmer in kmerlist:
        pwm += np.array(list(kmer))[:,None] == np.array(list('ACGT'))
    pwm /= np.sum(pwm, axis = 1)[:,None]
    return pwm
        
def reverse_complement(kmer):
    kmer = np.array(list(kmer))[::-1]
    nkmer = np.copy(kmer)
    nkmer[kmer =='A'] = 'T'
    nkmer[kmer =='C'] = 'G'
    nkmer[kmer =='G'] = 'C'
    nkmer[kmer =='T'] = 'A'
    return ''.join(nkmer)




dna_motifs = 60 # number of tfs
rna_motifs = 40 # number of rbps


l_motifs = 7 # median length of tf and rbp motifs

# array with length of tf_motifs
len_tf_motifs = np.random.normal(loc = 0, scale = 1.1, size = dna_motifs)
len_tf_motifs[len_tf_motifs>0] =  len_tf_motifs[len_tf_motifs>0]**2
len_tf_motifs = l_motifs + len_tf_motifs.astype(int)

# array with length of rbp_motifs
len_rbp_motifs = np.random.normal(loc = 0, scale = 1.1, size = rna_motifs)
len_rbp_motifs[len_rbp_motifs>0] =  len_rbp_motifs[len_rbp_motifs>0]**2
len_rbp_motifs = l_motifs + len_rbp_motifs.astype(int) 

tf_has_context = 0.5 # fraction of tfs with duplex context specificity 
rbp_has_context = 0.5 # fraction of rbps with duplex context specificity

motifcomplex = False

n_seqs = 100 # number of sequences to generate

l_enhancer = 100 # lengths of enhancers
l_promoter = 100 # length of promoters
l_5utr = 100 # length of 5utrs
l_cds = 500 # length of cds
l_3utr = 200 # length of 3utr
l_insulator = 200 # length of dna elements that have no function

# the resulting sequence length is 
l_seq = l_enhancer + l_insulator + l_promoter + l_5utr + l_cds + l_3utr + l_insulator + l_enhancer

# exp_func defines whether where looking at kt/kd or exp(kt-kd)
exp_func = False

multiply_enhancer = False # Defines whether enhancer and promoter activity are multiplied or summed up together

noiseratio = 0.25 # noise ratio, is the amount of gaussian noise to real variance in the data
ncondition = 81 # number of different cell types

present_all = 0.2 # fraction of motifs (tfs and rbps), that are always present
# should determine the average transcription rate
# these are all activators

subsample = 0.4 # fraction of tfs that are present in a cell type

fraction_of_activators = 0.7 # 70% of tfs are activators, 30% are repressors
fraction_of_stabilizers = 0.3 # 30% of rbps stabilize transcripts, 30% increase degradations


enrichseqs = 2 # add number of 'enrichseqs' motifs to regulatory regions in sequence

activator_interactions = 0.3 # which fraction of activators has a partner that increases activation even further if motif in the same region
stabilizer_interactions = 0.2 # which fraction of transcript stabilizers has a partner that increases activation even further if motif in the same region

outname = 'Sim'+str(n_seqs)+'l'+str(l_enhancer)+'-'+str(l_promoter)+'-'+str(l_5utr)+'-'+str(l_cds)+'-'+str(l_3utr)+'-'+str(l_insulator)+'m'+str(dna_motifs)+'-'+str(rna_motifs)+'-'+str(l_motifs)+str(motifcomplex)[0]+'n'+str(noiseratio)+'c'+str(ncondition)+'p'+str(present_all)+'s'+str(subsample)+'f'+str(fraction_of_activators)+'-'+str(fraction_of_stabilizers)+'a'+str(activator_interactions)+'-'+str(stabilizer_interactions)+'e'+str(exp_func)[0]+'r'+str(enrichseqs)+'M'+str(multiply_enhancer)[0]+'x'+str(tf_has_context)+str(rbp_has_context)
print(outname)


# indexes of activators and repressors
activators = np.random.permutation(dna_motifs)[:int(dna_motifs*fraction_of_activators)]
repressors = np.arange(dna_motifs, dtype = int)[~np.isin(np.arange(dna_motifs, dtype = int),activators)]
tf_effect = np.ones(dna_motifs)
tf_effect[repressors] = -1 # activators have positive effect and repressors have negative effect

stabilizers = np.random.permutation(rna_motifs)[:int(rna_motifs*fraction_of_stabilizers)]
degraders = np.arange(rna_motifs, dtype = int)[~np.isin(np.arange(rna_motifs, dtype = int),stabilizers)]
rbp_effect = np.ones(rna_motifs)
rbp_effect[stabilizers] = -1 # stabilizers have negative or small degradation rate while degraders have positive degradation rate effect

# define set of universal tfs and rbps
n_act, n_rep = int(np.ceil(len(activators)*present_all)), int(np.ceil(len(repressors)*present_all))
universal_tfs = np.append(activators[:n_act], repressors[:n_rep])
specific_tfs = np.append(activators[n_act:], repressors[n_rep:])

n_stab, n_deg = int(np.ceil(len(stabilizers)*present_all)), int(np.ceil(len(degraders)*present_all))
universal_rbps = np.append(stabilizers[:n_stab], degraders[:n_deg])
specific_rbps = np.append(stabilizers[n_stab:], degraders[n_deg:])

# define set of tfs and rbps per cell type
cell_tfs = []
cell_rbps = []
for nc in range(ncondition):
    ctfs = np.random.permutation(specific_tfs)[:int(np.ceil(len(specific_tfs)*subsample))]
    cell_tfs.append(np.append(universal_tfs, ctfs))
    crbps = np.random.permutation(specific_rbps)[:int(np.ceil(len(specific_rbps)*subsample))]
    cell_rbps.append(np.append(universal_rbps, crbps))

# set of activators that interact positively if they occur in the same region, or in promoter and other region
if activator_interactions > 0:
    interacting_activators = np.sort(np.random.permutation(activators)[:int(len(activators)*activator_interactions)])
# set of stabilizers that interact positively if they both occur in the 3'utr
if stabilizer_interactions >0:
    interacting_stabilizers = np.sort(np.random.permutation(stabilizers)[:int(len(stabilizers)*stabilizer_interactions)])


# Effect strenth of transcription factors
tf_effect_size = (1.+9*np.random.random(dna_motifs))*tf_effect
rbp_effect_size = (1.+9*np.random.random(rna_motifs))*rbp_effect

# each context duplet (GC, CC, etc) adds this to the tf effect of binding, the maximum length of simulated context is 15 and 20
tmaxcont = 15
rmaxcont = 20
tf_context_effect = tf_effect_size/(2.*tmaxcont)
rbp_context_effect = rbp_effect_size/(2.*rmaxcont)

# additional effect strength of activators and stabilizers that interact with others
if activator_interactions > 0:
    max_effect_boost = 3. # additional effect is at least as much as the maximum effect of the two tfs and at the most 3 times the effect of the max effect of the two tfs
    tf_interaction_effect = np.array([tf_effect_size for i in range(len(tf_effect_size))])
    tf_interaction_effect = np.amax(np.array([tf_interaction_effect, tf_interaction_effect.T]), axis = 0) * ((max_effect_boost-1.)*np.random.random(size = np.shape(tf_interaction_effect))+1.)
    tf_interaction_effect[~np.isin(np.arange(len(tf_interaction_effect)), interacting_activators)] = 0
    tf_interaction_effect[:,~np.isin(np.arange(len(tf_interaction_effect)), interacting_activators)] = 0
    tf_interaction_effect = np.triu(tf_interaction_effect, k= 1)
    
# additional effect strength of activators and stabilizers
if stabilizer_interactions > 0:
    max_effect_boost = 3. # additional effect is at least as much as the maximum effect of the two tfs and at the most 3 times the effect of the max effect of the two tfs
    rbp_interaction_effect = np.array([rbp_effect_size for i in range(len(rbp_effect_size))])
    rbp_interaction_effect = np.amax(np.array([rbp_interaction_effect, rbp_interaction_effect.T]), axis = 0) * ((max_effect_boost-1.)*np.random.random(size = np.shape(rbp_interaction_effect))+1.)
    rbp_interaction_effect[~np.isin(np.arange(len(rbp_interaction_effect)), interacting_stabilizers)] = 0
    rbp_interaction_effect[:,~np.isin(np.arange(len(rbp_interaction_effect)), interacting_stabilizers)] = 0
    rbp_interaction_effect = np.triu(rbp_interaction_effect,k = 1)


    
tf_motifs = []
rbp_motifs = []
# generate motifs of tfs and rbps
i = 0
while i < dna_motifs:
    nmotif = ''.join(np.random.choice(['A', 'C', 'G', 'T',], size = len_tf_motifs[i]))
    if nmotif not in tf_motifs:
        tf_motifs.append(nmotif)
        i += 1

# create two nt context for motifs from some intrinsic position in motif
tf_has_context = np.random.permutation(dna_motifs)[:int(dna_motifs*tf_has_context)] # tf_has_context was set to 0 then no motif will have context
rbp_has_context = np.random.permutation(rna_motifs)[:int(rna_motifs*rbp_has_context)] #rbp_has_context was set to 0 then no motif will have context
# only allow some common patterns to be context, not all duplets
possible_dna_context = np.array(['GC', 'CG', 'TA', 'AT'])
possible_rna_context = np.array(['CC', 'GG', 'GC', 'CG', 'TT', 'TG', 'GT'])

tf_context = []
tf_context_reverse = []
for t, tfm in enumerate(tf_motifs):
    if t in tf_has_context:
        mask = np.array([dn in tfm for dn in possible_dna_context])
        if not mask.any():
            mask[0] = True
        tf_context.append(np.random.choice(possible_dna_context[mask]))
        tf_context_reverse.append(reverse_complement(tf_context[-1]))
    else:
        tf_context.append('')
        tf_context_reverse.append('')

# Next steps, vary effect slightly for varied motifs
# add nm (between min_specificity and max_specificity) additional variations of the motif to the set
exptf_motifs = []
tf_ids = []
for i, tfmotif in enumerate(tf_motifs):
    exptf_motifs.append(tfmotif)
    tf_ids.append(i)
    max_specificity = max(1,int((len_tf_motifs[i]-5)*1.5)) # minimum number of k-mers with alterations at one base that can be assigned to a motif
    min_specificity = int(len_tf_motifs[i] *1.7) # maximum k-mers with alterations at one base that can be assigned to a motif
    nm = np.random.randint(max_specificity-1, min_specificity-1) # number of k-mers with single base-pair changes for this motif
    # get all allowed changes
    w = np.where(np.array(list(tfmotif))[:,None] != nuctides)
    if not motifcomplex:
        nm = len(w[0])
    potc = np.random.permutation(w[0]*4+w[1])[:nm]
    for psal in potc:
        ntfmotif = str(tfmotif)
        ntfmotif = ntfmotif[:int(psal/4)] + nuctides[psal%4]+ ntfmotif[int(psal/4)+1:]
        exptf_motifs.append(ntfmotif)
        tf_ids.append(i)
        
tf_motifs = np.array(exptf_motifs)
tf_motifs_reverse = np.array([reverse_complement(tfm) for tfm in tf_motifs])
tf_ids = np.array(tf_ids, dtype = int)
# 


i = 0
while i < rna_motifs:
    nmotif = ''.join(np.random.choice(['A', 'C', 'G', 'T',], size = len_rbp_motifs[i]))
    if nmotif not in rbp_motifs:
        rbp_motifs.append(nmotif)
        i += 1

rbp_context = []
rbp_context_reverse = []
for t, rfm in enumerate(rbp_motifs):
    if t in rbp_has_context:
        mask = np.array([dn in rfm for dn in possible_rna_context])
        if not mask.any():
            mask[0] = True
        rbp_context.append(np.random.choice(possible_rna_context[mask]))
        rbp_context_reverse.append(reverse_complement(rbp_context[-1]))
    else:
        rbp_context.append('')
        rbp_context_reverse.append('')

# add nm (between min_specificity and max_specificity) additional variations of the motif to the set
exprbp_motifs = []
rbp_ids = []
for i, rbmotif in enumerate(rbp_motifs):
    exprbp_motifs.append(rbmotif)
    rbp_ids.append(i)
    max_specificity = max(1,int((len_rbp_motifs[i]-5)*1.5)) # minimum number of k-mers with alterations at one base that can be assigned to a motif
    min_specificity = int(len_rbp_motifs[i] *1.7) # maximum k-mers with alterations at one base that can be assigned to a motif
    nm = np.random.randint(max_specificity-1, min_specificity-1)
    w = np.where(np.array(list(rbmotif))[:,None] != nuctides)
    if not motifcomplex:
        nm = len(w[0])
    potc = np.random.permutation(w[0]*4+w[1])[:nm]
    for psal in potc:
        nrbmotif = str(rbmotif)
        nrbmotif = nrbmotif[:int(psal/4)] + nuctides[psal%4]+ nrbmotif[int(psal/4)+1:]
        exprbp_motifs.append(nrbmotif)
        rbp_ids.append(i)
        
rbp_motifs = np.array(exprbp_motifs)
rbp_ids = np.array(rbp_ids, dtype = int)

# compute the pwms for the intrinsic specificities from the kmers
tpwms = [kmer_to_pwm(tf_motifs[tf_ids == k]) for k in np.unique(tf_ids)]
rpwms = [kmer_to_pwm(rbp_motifs[rbp_ids == k]) for k in np.unique(rbp_ids)]


# This generates variable insulator lengths
insulater_length = np.random.randint(0, l_insulator, size = (n_seqs,2))
# This generates variable cds lenghts
cds_length = np.random.randint(0, l_cds/2, size = n_seqs)

# This is a matrix swill all the regional lengths
seqmat = np.zeros((n_seqs, 8), dtype = int)
seqmat[:,0] = l_enhancer
seqmat[:,1] = l_insulator - insulater_length[:,0]
seqmat[:,2] = l_promoter
seqmat[:,3] = l_5utr
seqmat[:,4] = l_cds - cds_length
cdscorrect = seqmat[:,4]%3
cds_length -= cdscorrect
seqmat[:,4] -= cdscorrect 
seqmat[:,5] = l_3utr
seqmat[:,6] = l_insulator - insulater_length[:,1]
seqmat[:,7] = l_enhancer

tsss = np.sum(seqmat[:,:3], axis = 1)
stcdn = np.sum(seqmat[:,:4], axis = 1)
stpcdn = np.sum(seqmat[:,:5], axis = 1)
clste = np.sum(seqmat[:,:6], axis = 1)


if '--realistic_sequence_distribution' in sys.argv:
    # load realistic triplet frequencies and sample from them.
    outname += '_realseq'
    random_triplets, rprob, rconprob = read_probs(sys.argv[sys.argv.index('--realistic_sequence_distribution')+1])
    promoter_triplets, pprob, pconprob = read_probs(sys.argv[sys.argv.index('--realistic_sequence_distribution')+2])
    enhancer_triplets, eprob, econprob = read_probs(sys.argv[sys.argv.index('--realistic_sequence_distribution')+3])
    codon_triplets, cprob, cconprob = read_probs(sys.argv[sys.argv.index('--realistic_sequence_distribution')+4])
    utr3_triplets, u3prob, u3conprob = read_probs(sys.argv[sys.argv.index('--realistic_sequence_distribution')+5])
    utr5_triplets, u5prob, u5conprob = read_probs(sys.argv[sys.argv.index('--realistic_sequence_distribution')+6])
    
    probs = [[enhancer_triplets, eprob, econprob, False], [random_triplets, rprob, rconprob, False], [promoter_triplets, pprob, pconprob, False], [utr5_triplets, u5prob, u5conprob, False],[codon_triplets, cprob, cconprob, True], [utr3_triplets, u3prob, u3conprob, False], [random_triplets, rprob, rconprob, False], [enhancer_triplets, eprob, econprob, False]]
    
    t0 = time.time()
    sequences = sample_sequences(seqmat, probs)
    t1 = time.time()
    print(t1-t0)
else:
    ##### Generate random sequences
    sequences = []
    for s in range(n_seqs):
        sequences.append(''.join(np.random.choice(['A', 'C', 'G', 'T',], size = l_seq)))



def get_cont_motif(tf_motifs, tf_ids, tf_has_context, tf_context, max_context = 25):
    
    pro_en = np.random.randint(0, len(tf_motifs))
    pro_id = tf_ids[pro_en]
    
    ##### if the sampled motif has a context preference ### Add this the the other regions as well!
    if pro_id in tf_has_context:
        # randomly determine context length between 0,25 duplets on  from exponential distribution
        context_len = np.exp(np.random.random()*np.log(max_context+1)).astype(int)-1
        context = ''
        for i in range(context_len):
            context += tf_context[pro_id]
        context_left = np.random.randint(context_len+1)*2
        tfm = context[:context_left] + tf_motifs[pro_en] + context[context_left:]
    else:
        tfm = tf_motifs[pro_en]
        context_len = 0
    return tfm, context_len, pro_en
    
# There is no effect of TF binding orientation: https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-016-2549-x
def insert_random_strand(tfm, pro_loc, seq):
    # decide randomly whether to insert motif in forward or reverse strand.
    if np.random.randint(2):
        seq = seq[:pro_loc] + reverse_complement(tfm) + seq[pro_loc+len(tfm):]
    else:
        seq = seq[:pro_loc] + tfm + seq[pro_loc+len(tfm):]
    return seq

# just to check if motifs are put in there correctly
#for s, seq in enumerate(sequences):
    #sequences[s] = seq.lower()
    
# enrich functional sequences with generated motifs
if enrichseqs>0:
    for s, seq in enumerate(sequences):
        ins0, ins1 = insulater_length[s]
        cdsv = cds_length[s]
        for e in range(enrichseqs):
            
            # enrich promter sequence with tf motifs and or its reverse complement
            tfm, context_len, pro_en = get_cont_motif(tf_motifs, tf_ids, tf_has_context, tf_context, max_context = tmaxcont)
            pro_loc = np.random.randint(l_enhancer+l_insulator-ins0, l_enhancer+l_insulator-ins0+l_promoter-(len_tf_motifs[tf_ids[pro_en]]+context_len)+1)
            sequences[s] = insert_random_strand(tfm, pro_loc, sequences[s])
            
            
            # enrich first enhancer
            tfm, context_len, pro_en = get_cont_motif(tf_motifs, tf_ids, tf_has_context, tf_context, max_context = tmaxcont)
            pro_loc =np.random.randint(0, l_enhancer-(len_tf_motifs[tf_ids[pro_en]]+context_len)+1)
            sequences[s] = insert_random_strand(tfm, pro_loc, sequences[s])
            
            # enrich second enhancer
            tfm, context_len, pro_en = get_cont_motif(tf_motifs, tf_ids, tf_has_context, tf_context, max_context = tmaxcont)
            pro_loc = np.random.randint(l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv+l_3utr+l_insulator-ins1, l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv+l_3utr+l_insulator-ins1+l_enhancer-(len_tf_motifs[tf_ids[pro_en]]+context_len)+1)
            sequences[s] = insert_random_strand(tfm, pro_loc, sequences[s])
            
            # enrich 3utr
            tfm, context_len, pro_en = get_cont_motif(rbp_motifs, rbp_ids, rbp_has_context, rbp_context, max_context = rmaxcont)
            pro_loc = np.random.randint(l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv, l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv+l_3utr-(len_rbp_motifs[rbp_ids[pro_en]]+context_len)+1)
            sequences[s] = sequences[s][:pro_loc] + tfm + sequences[s][pro_loc+len(tfm):]


    
if '--realistic_sequences' in sys.argv:
    outname += 'smots'
    # add sequence well know sequence patterns
    sequences = add_sequencemarks(sequences, ['GGGCGCC', 'GGCCGCC', 'GCCCGCC', 'CGGCGCC', 'GCGCGCC', 'CGCCGCC', 'CCGCGCC', 'CCCCGCC'], tsss- np.random.choice(np.arange(35,55, dtype = int), n_seqs))
    sequences = add_sequencemarks(sequences, 'TATAA', tsss - np.random.choice(np.arange(25, 35, dtype = int), n_seqs))
    sequences = add_sequencemarks(sequences, ['CCACC','CCGCC', 'GCACC', 'CCAAC', 'CCGAC'], stcdn-5)
    t2 = time.time()
    sequences = add_sequencemarks(sequences, 'AUG', stcdn)
    sequences = add_sequencemarks(sequences, ['UAG', 'UAA', 'UGA'],stpcdn)
    sequences = add_sequencemarks(sequences, ['AATAAA', 'ATTAAA'], clste -np.random.choice(np.arange(20,50, dtype =int), n_seqs))

def printseq(seq, inc = 50):
    for i in range(int(len(seq)/50)):
        print(seq[i*50:(i+1)*50], 50*(i+1))
'''    
printseq(sequences[0])
printseq(sequences[1])
printseq(sequences[2])
sys.exit()
'''

# introduce a max number all factors after which effect cannot be multiplied by the number of occurrances anymore
# scan sequence forward and reverse complement
# if overlap of motifs occurs, the stronger factor occupies the location +- 2 in the sequence
# factors need to be expressed in the cell type to occupy the space
# For each individual length present in the motif set: Pick a subsequence and check if in the set of motifs, assign ones/len_motif to all positions of the motif --> Generate matrix n_motifsXn_positions
    # for each cell type set all the rows with inactive factors to zero
    # add a non-linear motif interactions to motifs-locations that interact but don't overlap
    # start from motif with highest effect and remove all motifs that are within the same region


def calc_ncontext(lseq, rseq, cont, lc):
    cl, cr = 0, 0
    for l in range(len(lseq)-lc, -1, -lc):
        if lseq[l:l+lc] == cont:
            cl += lc
        else:
            break
    for l in range(0,len(rseq), lc):
        if rseq[l:l+lc] == cont:
            cr += lc
        else:
            break
    return cl, cr

def scan_sequence(seq, len_motifs, ids, features, motifs, context, contextratio, motifs_reverse = None, context_reverse = None):
    
    for lmot in np.unique(len_motifs):
        for l in range(len(seq)-lmot+1):
            cmot = seq[l:l+lmot]
            if cmot in motifs:
                mask = np.where(np.isin(motifs, cmot))[0]
                features[mask,l:l+lmot] = 1
                idss = ids[mask]
                for i, ii in enumerate(idss):
                    mcont = context[ii]
                    if mcont != '':
                        conl, conr = calc_ncontext(seq[:l], seq[l+lmot:], mcont, 2)
                        features[mask[i],l:l+lmot] += (conl+conr)/contextratio
                        
            if motifs_reverse is not None:
                if cmot in motifs_reverse:
                    mask = np.where(np.isin(motifs_reverse, cmot))[0]
                    features[mask,l:l+lmot] = 1
                    idss = ids[mask]
                    for i, ii in enumerate(idss):
                        mcont = context_reverse[ii]
                        if mcont != '':
                            conl, conr = calc_ncontext(seq[:l], seq[l+lmot:], mcont, 2)
                            features[mask[i],l:l+lmot] += (conl+conr)/contextratio
                        
    return features

promoter_features = np.zeros((n_seqs, len(tf_motifs), l_promoter))
enhancer_features = np.zeros((n_seqs,2, len(tf_motifs), l_enhancer))
rna3_features = np.zeros((n_seqs, len(rbp_motifs), l_3utr))

t0 = time.time()
for s, seq in enumerate(sequences):
    
    if s% 10 == 1:
        print(s, time.time()-t0)
        t0 = time.time()
    # compute variable length of insulator
    ins0, ins1 = insulater_length[s]
    cdsv = cds_length[s]
    
    # compute counts for promoter sequences
    pr_seq = seq[l_enhancer+l_insulator-ins0:l_enhancer+l_insulator-ins0+l_promoter]
    promoter_features[s] = scan_sequence(pr_seq, len_tf_motifs, tf_ids, promoter_features[s], tf_motifs, tf_context, 2*tmaxcont, motifs_reverse = tf_motifs_reverse, context_reverse = tf_context_reverse)
    
    # compute counts for rna regions
    rna_seq = seq[l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv: l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv+l_3utr]
    rna3_features[s] = scan_sequence(rna_seq, len_rbp_motifs, rbp_ids, rna3_features[s], rbp_motifs, rbp_context, 2*rmaxcont)
    
    # compute counts for enhancer0
    en_seq0 = seq[0 : l_enhancer]
    enhancer_features[s,0] = scan_sequence(en_seq0, len_tf_motifs, tf_ids, enhancer_features[s,0], tf_motifs, tf_context, 2*tmaxcont, motifs_reverse = tf_motifs_reverse, context_reverse = tf_context_reverse)
        
    # compute counts for enhancer1
    en_seq1 = seq[l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv+l_3utr+l_insulator-ins1 : l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv+l_3utr+l_insulator-ins1+l_enhancer]
    enhancer_features[s,1] = scan_sequence(en_seq0, len_tf_motifs, tf_ids, enhancer_features[s,1], tf_motifs, tf_context, 2*tmaxcont, motifs_reverse = tf_motifs_reverse, context_reverse = tf_context_reverse)




# reduce different kmers of same tf to single binding profile for each tf
unique_tf, uni_index = np.unique(tf_ids, return_index = True)
for i, uni in enumerate(unique_tf):
    promoter_features[:, uni_index[i]] = np.amax(promoter_features[:, tf_ids == uni], axis = 1)
promoter_features = promoter_features[:, uni_index]    

# reduce different kmers of same tf to single binding profile for each tf
unique_rbp, uni_index = np.unique(rbp_ids, return_index = True)
for i, uni in enumerate(unique_rbp):
    rna3_features[:, uni_index[i]] = np.amax(rna3_features[:, rbp_ids == uni], axis = 1)
rna3_features = rna3_features[:, uni_index]

# reduce different kmers of same tf to single binding profile for each tf
unique_tf, uni_index = np.unique(tf_ids, return_index = True)
for i, uni in enumerate(unique_tf):
    enhancer_features[:,0, uni_index[i]] = np.amax(enhancer_features[:,0, tf_ids == uni], axis = 1)
    enhancer_features[:,1, uni_index[i]] = np.amax(enhancer_features[:,1, tf_ids == uni], axis = 1)
enhancer_features = enhancer_features[:, :, uni_index]


# compute the distance between enhancer center to tss to weight the strength of elements in the enhancer. 
enhancer_to_tss = -insulater_length + np.array([l_enhancer/2+l_insulator+l_promoter, l_5utr+l_cds+l_3utr+l_insulator+l_enhancer/2])
enhancer_to_tss[:,1] - cds_length

# compute the expression for every cell type with active factors
celltypes = []
expression = np.zeros((n_seqs, ncondition))
degradation = np.zeros((n_seqs, ncondition))
transcription_real = np.zeros((n_seqs, ncondition))
for c in range(ncondition):
    celltypes.append('C'+str(c))
    
    active_tf_effect = tf_effect_size[cell_tfs[c]]
    # add activator interaction activity to tf_effect_size (no positional constraints for activators)
    # if interaction activity is added there is a higher chance that the two tfs outcompete other tfs in the same region
    # effect is added independent of relative positions of motifs -- only needs to be in the same region
    involved = None
    if activator_interactions > 0:
        # for potentially active interactions in this cell type, establish a multiplication matrix that multiplies sequences that contain both interactors with this effect
        active_interactions = tf_interaction_effect[cell_tfs[c]][:,cell_tfs[c]]
        if (active_interactions != 0).any():
            # select all interactions that have non-zero interaction strength
            involved = np.array(np.where(active_interactions!=0)).T
            # multipicative factors derived from tf_interaction_effect and tf_effect_size
            interaction_activity = np.ones(np.shape(involved))
            for i, inv in enumerate(involved):
                ct0, ct1, ai = inv[0], inv[1], active_interactions[inv[0], inv[1]]
                # Determine multipliers for each tf in that category
                interaction_activity[i,0] = (active_tf_effect[ct0] + ai/2)/active_tf_effect[ct0]
                interaction_activity[i,1] =(active_tf_effect[ct1] + ai/2)/active_tf_effect[ct1]
    
    active_promoter = promoter_features[:,cell_tfs[c]]
    if involved is not None:
        for i, ct in enumerate(involved):
            ct0, ct1 = ct
            # mark all sequences that have this interaction
            active_intinseq = np.prod(np.sum(active_promoter[:,[ct0,ct1]], axis = 2) != 0, axis = 1) > 0
            # multiply these sequences with the multiplier
            active_promoter[active_intinseq, ct0] *= interaction_activity[i,0]
            active_promoter[active_intinseq, ct1] *= interaction_activity[i,1]
            
    # multiply the occurance with the effect
    active_promoter = active_promoter * active_tf_effect[None,:,None]
    
    # resolve overlapping motifs
        # if two motifs are within 2bp then only keep the footprint of the stronger motif
    for i in range(np.shape(active_promoter)[-1]):
        maxactive = np.amax(np.absolute(active_promoter[:,:,i]),axis = -1)
        mask = np.absolute(active_promoter[:,:,i]) != maxactive[:,None]
        active_promoter[:,:,max(0,i-2):min(l_promoter, i+2)][mask] = 0
        
    # restrict binding sites to three full length footprints, similar to sigmoid functoin for binding reactions
    active_nonzero = np.sum(active_promoter !=0, axis = -1)/len_tf_motifs[cell_tfs[c]]
    promoter_effect = np.sum(active_promoter,axis = -1)/len_tf_motifs[cell_tfs[c]]
    if (active_nonzero > 3).any():
        promoter_effect[active_nonzero > 3] *3./active_nonzero[active_nonzero > 3] 
        
    
    
    
    
    # SAME for enhancers
    active_enhancer = enhancer_features[:,:,cell_tfs[c]]
    if involved is not None:
        for i, ct in enumerate(involved):
            ct0, ct1 = ct
            for j in range(2):
                # mark all sequences that have this interaction
                active_intinseq = np.prod(np.sum(active_enhancer[:,j,[ct0,ct1]], axis = -1) != 0, axis = -1) > 0
                # multiply these sequences with the multiplier
                active_enhancer[active_intinseq, j, ct0] *= interaction_activity[i,0]
                active_enhancer[active_intinseq, j, ct1] *= interaction_activity[i,1]
    
    
    # resolve overlapping motifs
    active_enhancer = active_enhancer * active_tf_effect[None,None,:,None]
    for i in range(np.shape(active_enhancer)[-1]):
        maxactive = np.amax(np.absolute(active_enhancer[:,:,:,i]),axis = -1)
        active_enhancer[:,:,:,max(0,i-2):min(l_enhancer, i+2)][np.absolute(active_enhancer[:,:,:,i]) != maxactive[...,None]] = 0
    # restrict binding sites to two full length footprints, similar to sigmoid functoin for binding reactions
    active_nonzero = np.sum(active_enhancer !=0, axis = -1)/len_tf_motifs[cell_tfs[c]]
    enhancer_effect = np.sum(active_enhancer,axis = -1)/len_tf_motifs[cell_tfs[c]]
    if (active_nonzero > 3).any():
        enhancer_effect[active_nonzero > 3] *3./active_nonzero[active_nonzero > 3] 
    
    
    
    
    # AND FINALLY for RNA
    active_rbp_effect = rbp_effect_size[cell_rbps[c]]
    
    involved = None
    if stabilizer_interactions > 0:
        # for potentially active interactions in this cell type, establish a multiplication matrix that multiplies sequences that contain both interactors with this effect
        active_interactions = rbp_interaction_effect[cell_rbps[c]][:,cell_rbps[c]]
        if (active_interactions != 0).any():
            # select all interactions that have non-zero interaction strength
            involved = np.array(np.where(active_interactions!=0)).T
            # multipicative factors derived from tf_interaction_effect and tf_effect_size
            interaction_activity = np.ones(np.shape(involved))
            for i, inv in enumerate(involved):
                ct0, ct1, ai = inv[0], inv[1], active_interactions[inv[0], inv[1]]
                # Determine multipliers for each tf in that category
                interaction_activity[i,0] = (active_rbp_effect[ct0] + ai/2)/active_rbp_effect[ct0]
                interaction_activity[i,1] =(active_rbp_effect[ct1] + ai/2)/active_rbp_effect[ct1]
    
    
    # resolve overlapping motifs
    active_rna = rna3_features[:,cell_rbps[c]]
    if involved is not None:
        for i, ct in enumerate(involved):
            ct0, ct1 = ct
            # mark all sequences that have this interaction
            active_intinseq = np.prod(np.sum(active_rna[:,[ct0,ct1]], axis = 2) != 0, axis = 1) > 0
            # multiply these sequences with the multiplier
            active_rna[active_intinseq, ct0] *= interaction_activity[i,0]
            active_rna[active_intinseq, ct1] *= interaction_activity[i,1]
    
    active_rna = active_rna * active_rbp_effect[None,:,None]
    for i in range(np.shape(active_rna)[-1]):
        maxactive = np.amax(np.absolute(active_rna[:,:,i]),axis = -1)
        active_rna[:,:,max(0,i-2):min(l_3utr, i+2)][np.absolute(active_rna[:,:,i]) != maxactive[...,None]] = 0
    # restrict binding sites to two full length footprints, similar to sigmoid functoin for binding reactions
    active_nonzero = np.sum(active_rna !=0, axis = -1)/len_rbp_motifs[cell_rbps[c]]
    # sum over all positions
    rna_effect = np.sum(active_rna, axis = -1)/len_rbp_motifs[cell_rbps[c]]
    if (active_nonzero > 2).any():
        rna_effect[active_nonzero > 2] *2./active_nonzero[active_nonzero > 2] 
    
    
    promoter_effect, enhancer_effect, rna_effect = np.sum(promoter_effect, axis = -1), np.sum(enhancer_effect, axis = -1), np.sum(rna_effect, axis = -1)
    
    cexpression, ctranscription, cdegradation = compute_expression(promoter_effect, enhancer_effect, enhancer_to_tss, rna_effect, exp_func = False, multiply_enhancer = multiply_enhancer)
    expression[:,c] = cexpression
    degradation[:,c] = cdegradation
    transcription_real[:,c] = ctranscription



'''
# old code for kmer counts
# count tf features in enhancers and promoters 
promoter_features = np.zeros((n_seqs, dna_motifs))
enhancer_features = np.zeros((n_seqs,2, dna_motifs))
rna3_features = np.zeros((n_seqs, rna_motifs))
for s, seq in enumerate(sequences):
    # compute variable length of insulator
    ins0, ins1 = insulater_length[s]
    cdsv = cds_length[s]
    
    # compute counts for promoter sequences
    pr_seq = seq[l_enhancer+l_insulator-ins0:l_enhancer+l_insulator-ins0+l_promoter]
    sfeat = []
    for l in range(l_promoter-l_motifs+1):
        sfeat.append(pr_seq[l:l+l_motifs])
    us, ns = np.unique(sfeat, return_counts = True)
    promoter_features[s, np.isin(tf_motifs, us)] = ns[np.isin(us, tf_motifs)]
    #print('Promoter')
    #print(np.sum(promoter_features[s]), len(us))
    
    rna_seq = seq[l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv: l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv+l_3utr]
    sfeat = []
    for l in range(l_3utr-l_motifs+1):
        sfeat.append(rna_seq[l:l+l_motifs])
    us, ns = np.unique(sfeat, return_counts = True)
    rna3_features[s, np.isin(rbp_motifs, us)] = ns[np.isin(us, rbp_motifs)]
    #print(np.sum(rna3_features[s]), len(us))
    
    en_seq0 = seq[0 : l_enhancer]
    sfeat = []
    for l in range(l_enhancer-l_motifs+1):
        sfeat.append(en_seq0[l:l+l_motifs])
    us, ns = np.unique(sfeat, return_counts = True)
    enhancer_features[s,0,np.isin(tf_motifs, us)] = ns[np.isin(us, tf_motifs)]
    #print(enhancer_features[s,0])
    
    en_seq1 = seq[l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv+l_3utr+l_insulator-ins1 : l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv+l_3utr+l_insulator-ins1+l_enhancer]
    sfeat = []
    for l in range(l_enhancer-l_motifs+1):
        sfeat.append(en_seq1[l:l+l_motifs])
    us, ns = np.unique(sfeat, return_counts = True)
    enhancer_features[s,1,np.isin(tf_motifs, us)] = ns[np.isin(us, tf_motifs)]
    #print(enhancer_features[s,1])
    
enhancer_to_tss = -insulater_length + np.array([l_enhancer/2+l_insulator+l_promoter, l_5utr+l_cds+l_3utr+l_insulator+l_enhancer/2])
enhancer_to_tss[:,1] - cds_length

celltypes = []
expression = np.zeros((n_seqs, ncondition))
degradation = np.zeros((n_seqs, ncondition))
transcription_real = np.zeros((n_seqs, ncondition))
for c in range(ncondition):
    celltypes.append('C'+str(c))
    promoter_effect = np.sum(promoter_features[:,cell_tfs[c]] * tf_effect_size[cell_tfs[c]], axis = -1)
    enhancer_effect = np.sum(enhancer_features[:,:,cell_tfs[c]] * tf_effect_size[cell_tfs[c]], axis = -1)
    rna_effect = np.sum(rna3_features[:,cell_rbps[c]] * rbp_effect_size[cell_rbps[c]], axis = -1)
    
    if activator_interactions > 0:
        promoter_effect += np.sum(non_linear_feature(promoter_features)[:, non_linear_set(cell_tfs[c])] * tf_nonlin_effect_size[non_linear_set(cell_tfs[c])], axis = -1)
        enhancer_effect += np.sum(non_linear_feature(enhancer_features)[:, non_linear_set(cell_tfs[c])] * tf_nonlin_effect_size[non_linear_set(cell_tfs[c])], axis = -1)
    if stabilizer_interactions > 0:
        rna_effect += np.sum(non_linear_feature(rna3_features)[:, non_linear_set(cell_rbs[c])] * rn_nonlin_effect_size[non_linear_set(cell_rbps[c])], axis = -1)
    
    
    cexpression, ctranscription, cdegradation = compute_expression(promoter_effect, enhancer_effect, enhancer_to_tss, rna_effect, exp_func = False, multiply_enhancer = multiply_enhancer)
    expression[:,c] = cexpression
    degradation[:,c] = cdegradation
    transcription_real[:,c] = ctranscription
'''




expression_real = np.copy(expression)
degradation_real = np.copy(degradation)
    
# compute a gene specific noise std
expression_std = np.nan_to_num(np.std(expression, axis = 1))
expression_std[expression_std == 0] += np.amin(expression_std[expression_std != 0])
expression += noiseratio * np.random.normal(loc = 0., scale = expression_std, size = np.shape(expression.T)).T
expression[expression < 0] = 0

# compute a gene specific noise std
degradation_std = np.nan_to_num(np.std(degradation, axis = 1))
degradation_std[degradation_std == 0] += np.amin(degradation_std[degradation_std != 0])
degradation+= noiseratio * np.random.normal(loc = 0., scale = degradation_std, size = np.shape(degradation.T)).T
degradation[degradation < 0] = 0


expression_real = np.around(expression_real, 3)
expression = np.around(expression, 3)
degradation_real = np.around(degradation_real, 3)
degradation = np.around(degradation, 3)
# derive transcription rate from noisy measurements of degradation rate and noisy measurements of expression
transcription = expression * degradation
transcription= np.around(transcription, 3)



#### Generate all output files ###


tobj = open(outname+'tfpwms.txt', 'w')
for p, pwm in enumerate(tpwms):
    conspwm = ''.join(np.array(list('ACGT'))[np.argmax(pwm,axis=1)])
    tobj.write('Motif\tTF_'+str(p)+'\n#\t'+conspwm+' '+str(round(tf_effect_size[p],2))+'\nPos\tA\tC\tG\tT\n')
    for s in range(len(pwm)):
        tobj.write(str(s+1)+'\t'+str(round(pwm[s,0],3))+'\t'+str(round(pwm[s,1],3))+'\t'+str(round(pwm[s,2],3))+'\t'+str(round(pwm[s,3],3))+'\n')
    tobj.write('\n\n')

robj = open(outname+'rbppwms.txt', 'w')
for p, pwm in enumerate(rpwms):
    conspwm = ''.join(np.array(list('ACGT'))[np.argmax(pwm,axis=1)])
    robj.write('Motif\tRBP_'+str(p)+'\n#\t'+conspwm+' '+str(round(rbp_effect_size[p],2))+'\nPos\tA\tC\tG\tT\n')
    for s in range(len(pwm)):
        robj.write(str(s+1)+'\t'+str(round(pwm[s,0],3))+'\t'+str(round(pwm[s,1],3))+'\t'+str(round(pwm[s,2],3))+'\t'+str(round(pwm[s,3],3))+'\n')
    robj.write('\n\n')

genenames = np.array([['Gene_'+str(s+1)] for s in range(len(expression))])
np.savetxt(outname + '_exp.csv', np.append(genenames,expression_real, axis = 1), fmt='%s', delimiter = ',', header = ','.join(np.array(celltypes)))
np.savetxt(outname + '_expnsy.csv', np.append(genenames, expression, axis = 1), fmt='%s', delimiter = ',', header = ','.join(np.array(celltypes)))
np.savetxt(outname + '_deg.csv', np.append(genenames,degradation_real, axis = 1), fmt='%s', delimiter = ',', header = ','.join(np.array(celltypes)))
np.savetxt(outname + '_degnsy.csv', np.append(genenames,degradation, axis = 1), fmt='%s', delimiter = ',', header = ','.join(np.array(celltypes)))
np.savetxt(outname + '_transcnsy.csv', np.append(genenames,transcription, axis = 1), fmt='%s', delimiter = ',', header = ','.join(np.array(celltypes)))


cvfile = open(outname+'_cv10.txt', 'w')
cvsize = int(n_seqs/10)
for i in range(10):
    cvfile.write('# Test '+str(i)+'\n'+' '.join(genenames.flatten()[i*cvsize:(i+1)*cvsize])+'\n')

obj = open(outname+'_mrnas.fasta', 'w')
for s, seq in enumerate(sequences):
    ins0, ins1 = insulater_length[s]
    cdsv = cds_length[s]
    mrna_sequence = seq[l_enhancer+l_insulator-ins0+l_promoter : l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv+l_3utr]
    obj.write('>Gene_'+str(s+1)+'\n'+mrna_sequence+'\n')
obj.close()

# align DNA sequences on TSS for fasta file
tss_loc = -insulater_length[:,0] + l_enhancer+l_insulator+l_promoter
tss_to_end = l_seq - tss_loc
left_right_ext = max(np.amax(tss_loc), np.amax(tss_to_end))
obj = open(outname+'_dnas.fasta', 'w')
objl = open(outname+'_dnaslocation.txt', 'w')
for s, seq in enumerate(sequences):
    left_ext = ''.join(np.random.choice(['A', 'C', 'G', 'T',], size = left_right_ext - tss_loc[s]))
    right_ext = ''.join(np.random.choice(['A', 'C', 'G', 'T',], size = left_right_ext - tss_to_end[s]))
    obj.write('>Gene_'+str(s+1)+'\n'+left_ext+seq+right_ext+'\n')
    
    ins0, ins1 = insulater_length[s]
    cdsv = cds_length[s]
    
    objl.write('Gene_'+str(s+1)+' Enhancer '+str(len(left_ext))+'-'+str(len(left_ext)+l_enhancer)+' '+str(len(seq)+len(left_ext)+len(right_ext))+'\n'+
              'Gene_'+str(s+1)+' Enhancer '+str(len(left_ext)+l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv+l_3utr+l_insulator-ins1)+'-'+str(len(left_ext)+l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv+l_3utr+l_insulator-ins1+l_enhancer)+' '+str(len(seq)+len(left_ext)+len(right_ext))+'\n'+
              'Gene_'+str(s+1)+' Promoter '+str(len(left_ext)+l_enhancer+l_insulator-ins0)+'-'+str(len(left_ext)+l_enhancer+l_insulator-ins0+l_promoter)+' '+str(len(seq)+len(left_ext)+len(right_ext))+'\n'+
              'Gene_'+str(s+1)+' 5UTR '+str(len(left_ext)+l_enhancer+l_insulator-ins0+l_promoter)+'-'+str(len(left_ext)+l_enhancer+l_insulator-ins0+l_promoter+l_5utr)+' '+str(len(seq)+len(left_ext)+len(right_ext))+'\n'+
              'Gene_'+str(s+1)+' CDS '+str(len(left_ext)+l_enhancer+l_insulator-ins0+l_promoter+l_5utr)+'-'+str(len(left_ext)+l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv)+' '+str(len(seq)+len(left_ext)+len(right_ext))+'\n'+
              'Gene_'+str(s+1)+' 3UTR '+str(len(left_ext)+l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv)+'-'+str(len(left_ext)+l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv+l_3utr)+' '+str(len(seq)+len(left_ext)+len(right_ext))+'\n')
    #+l_enhancer+l_insulator-ins0+l_promoter+l_5utr+l_cds-cdsv+l_3utr+l_insulator-ins1+l_enhancer
    
obj.close()


# plot distribution of expression values with current simulation
if '--plot_expression' in sys.argv:
    from scipy.stats import gaussian_kde
    
    expnoise = expression_real + noiseratio * np.random.normal(loc = 0., scale = expression_std, size = np.shape(expression.T)).T
    # plot distribution of correlation between two expression files with different noise
    
    genecor = correlation(expnoise, expression, axis = 1, distance = False)
    cellcor = correlation(expnoise, expression, axis = 0, distance = False)
    
    fig0 = plt.figure(figsize = (3,3), dpi = 200)
    ax0 = fig0.add_subplot(111)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.boxplot([genecor, cellcor])
    ax0.set_xticks([1,2])
    ax0.set_xticklabels(['Genes', 'Celltypes'])
    ax0.set_ylabel('PCC noisy data')

    genecor = correlation(transcription, expression, axis = 1, distance = False)
    cellcor = correlation(transcription, expression, axis = 0, distance = False)
    
    fig = plt.figure(figsize = (3,3), dpi = 200)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.boxplot([genecor, cellcor])
    ax.set_xticks([1,2])
    ax.set_xticklabels(['Genes', 'Celltypes'])
    ax.set_ylabel('PCC Transcription - Expression')
    
    genecor = correlation(degradation, expression, axis = 1, distance = False)
    cellcor = correlation(degradation, expression, axis = 0, distance = False)
    
    fig1 = plt.figure(figsize = (3,3), dpi = 200)
    ax1 = fig1.add_subplot(111)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.boxplot([genecor, cellcor])
    ax1.set_xticks([1,2])
    ax1.set_xticklabels(['Genes', 'Celltypes'])
    ax1.set_ylabel('PCC Degradation - Expression')
    
    fig2 = plt.figure(figsize=(12.5,15.5), dpi = 80)
    for i in range(5):
        ax2 = fig2.add_subplot(6,5,i+1)
        ax2.hist(expression[:,c], bins = 30)
        ax2.set_yscale('log')
        ax2.set_title(celltypes[i])
        for j in range(5):
            vals = np.array([expression[:,i], expression[:,j]])
            if i != j:
                colors = np.log(1+gaussian_kde(vals)(vals))
            else:
                colors = np.ones(len(expression))
            ax3 = fig2.add_subplot(6,5,5+i+1+j*5)
            ax3.scatter(expression[:,i], expression[:,j], alpha = 0.2, c = colors, label='R='+str(round(pearsonr(expression[:,i], expression[:,j])[0],2)))
            ax3.set_xscale('symlog')
            ax3.set_yscale('symlog')
            ax3.legend()
    
    
    fig2.savefig(outname+'_expfor5.jpg', bbox_inches = 'tight', dpi = 200)
    fig.savefig(outname+'_expvstrans.jpg', bbox_inches = 'tight', dpi = 250)
    fig1.savefig(outname+'_expvsdeg.jpg', bbox_inches = 'tight', dpi = 250)
    fig0.savefig(outname+'_expreplicate.jpg', bbox_inches = 'tight', dpi = 250)
    #plt.show()

  
    
