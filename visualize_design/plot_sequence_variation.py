# plot_sequence variation
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from data_processing import readinfasta, quick_onehot, check
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from seqtofeature_beta import kmer_rep
from scipy.spatial.distance import cdist
from Levenshtein import distance as Lev_dist
from Bio import pairwise2
#from Bio.SubsMat import MatrixInfo as matlist
import Bio.Align.substitution_matrices as matlist
import umap
import sklearn.manifold as skm
from data_processing import check
from cnn_model import cnn
from generate_sequence import load_cnn_model
import torch

def sid(s1, s2):
    out = 0
    lens = len(s1)
    for i in range(lens):
        if s1[i] == s2[i]:
            out += 1.
        return 1.-out/float(lens)
        
def distance_wrap(x, y = None, distance = 'euclidean'):
    uptri = False
    if y is None:
        y = x
        uptri = True
    if distance == "edit" or distance == 'Levenshtein':
        dmat = np.zeros((len(x), len(y)))
        print(np.shape(dmat))
        for i,xi in enumerate(x):
            if uptri:
                low = i+1
            else:
                low = 0
            for j in range(low,len(y)):
                yi = y[j]
                dmat[i,j] = Lev_dist(xi, yi)
                if uptri:
                    dmat[j,i] = dmat[i,j]
    elif distance == 'identity' or distance == 'Identity' or distance == 'ID':
        dmat = np.zeros((len(x), len(y)))
        print(np.shape(dmat))
        for i,xi in enumerate(x):
            print(i)
            if uptri:
                low = i+1
            else:
                low = 0
            for j in range(low,len(y)):
                yi = y[j]
                alignment = pairwise2.align.localds(xi, yi, matlist.blosum62, -11,-1)
                dmat[i,j] = sid(alignment[0][0],alignment[0][1])
                if uptri:
                    dmat[j,i] = dmat[i,j]
    else:
        dmat = cdist(x, y, distance)
                
    return dmat
        

class embedding():
    def __init__(self, center = True, norm = False, distance = None, distance_embedding = False, regtype = 'kmer', representation = 'decreasing', reglen = 5, gaplength = 3, algorithm = 'svd', n_components = 2):
        self.distance = distance
        self.distance_embedding = distance_embedding
        self.representation = representation
        self.algorithm = algorithm
        self.regtype = regtype
        self.reglen = reglen
        self.gaplength = gaplength
        self.features = None
        if self.algorithm == 'NMF':
            self.center = False
            self.norm = False
        else:
            self.center = center
            self.norm = True
        self.mean = None
        self.std = None
        self.xfit = None
        self.n_components = n_components
        
    def seq_rep(self, x):
        
        if self.regtype == 'kmer': # and self.feature is None:
            x, kmers = kmer_rep(x, self.representation, self.reglen, self.gaplength, kmers = self.features)
            self.features = kmers
        
        elif self.regtype == 'CNN':
            x, nts = quick_onehot(x)
            x = torch.Tensor(x).transpose(1,2)
            batchsize = 100
            xrep = []
            with torch.no_grad():
                for i in range(0, len(x), batchsize):
                    xrep.append(self.representation.forward(x[i:i+batchsize], location = self.reglen).detach().cpu().numpy())
            xrep = np.concatenate(xrep,axis = 0)
            x = xrep
            print(np.shape(xrep))
        
        if self.center:
            if self.mean is None:
                self.mean = np.mean(x,axis = 0)
            x = x -self.mean
        
        if self.norm:
            if self.std is None:
                self.std = np.std(x, axis = 0)
                self.std[self.std == 0] = np.amax(self.std)
            x = x/self.std
        
        if self.distance_embedding:
            if self.xfit is None:
                self.xfit = x
            x = distance_wrap(x, self.xfit, distance = self.distance)
        
        return x
        
    def fit(self,x, **kwargs):
        
        if self.algorithm == 'TSNE' or self.algorithm == 'tsne' or self.algorithm == 'umap' or self.algorithm == 'UMAP':
            if 'metric' in kwargs.keys():
                if kwargs['metric'] == 'precomputed':
                    x = distance_wrap(x, distance = self.distance)
                metric = kwargs['metric']
                del kwargs['metric']
            else:
                metric = 'euclidean'
        
        if self.algorithm == 'svd' or self.algorithm == 'SVD':
            self.decoder = TruncatedSVD(n_components=self.n_components, n_iter=7, random_state=42)
        elif self.algorithm == 'NMF':
            self.decoder = NMF(n_components=self.n_components, init = 'nndsvda')
        elif self.algorithm == 'pca' or self.algorithm == 'PCA':
            self.decoder = PCA(n_components=self.n_components)
        elif self.algorithm == 'TSNE' or self.algorithm == 'tsne':
            print('Fit TSNE')
            self.decoder = skm.TSNE(n_components=self.n_components, metric = metric, **kwargs)
            # perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', metric_params=None, init='warn', verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='deprecated'
        elif self.algorithm == 'umap' or self.algorithm == 'UMAP':
            print('Fit UMAP')
            self.decoder = umap.UMAP(n_components=self.n_components, metric = metric, **kwargs)
            # n_neighbors=15, n_components=2, metric='euclidean', metric_kwds=None, output_metric='euclidean', output_metric_kwds=None, n_epochs=None, learning_rate=1.0, init='spectral', min_dist=0.1, spread=1.0, low_memory=True, n_jobs=-1, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0, negative_sample_rate=5, transform_queue_size=4.0, a=None, b=None, random_state=None, angular_rp_forest=False, target_n_neighbors=-1, target_metric='categorical', target_metric_kwds=None, target_weight=0.5, transform_seed=42, transform_mode='embedding', force_approximation_algorithm=False, verbose=False, tqdm_kwds=None, unique=False, densmap=False, dens_lambda=2.0, dens_frac=0.3, dens_var_shift=0.1, output_dens=False, disconnection_distance=None, precomputed_knn=(None, None, None)
        if self.algorithm == 'TSNE' or self.algorithm == 'tsne':
            return self.decoder.fit_transform(x)
        
        self.decoder.fit(x)
        print('Done fit')
        return self.transform(x)
    
    def transform(self,x):
        return self.decoder.transform(x)



if __name__ == '__main__':
    orig_names, orig_seqs = readinfasta(sys.argv[1])
    if '--random_set' in sys.argv:
        nrand = int(sys.argv[sys.argv.index('--random_set')+1])
        rand = np.random.permutation(len(orig_names))[:nrand]
        orig_seqs, orig_names = orig_seqs[rand], orig_names[rand]
    if '--selected_set' in sys.argv:
        selected_set = np.genfromtxt(sys.argv[sys.argv.index('--selected_set')+1], dtype = str)
        rand = np.isin(orig_names, selected_set)
        orig_seqs, orig_names = orig_seqs[rand], orig_names[rand]
        
    
    
    opt_names, opt_seqs = readinfasta(sys.argv[2])
    outname = os.path.splitext(sys.argv[2])[0]
    
    center = True
    if '--nocenter' in sys.argv:
        center = False
        outname += 'nocenter'
    
    if '--components' in sys.argv:
        components = np.array(sys.argv[sys.argv.index('--components')+1].split(','), dtype = int)
        outname += 'comp'+sys.argv[sys.argv.index('--components')+1]
    else:
        components = [0,1]
    
    
    regtype = 'kmer'
    kmertype = 'decreasing'
    kk = 5
    gap = 3
    if '--cnn' in sys.argv:
        regtype = 'CNN'
        predictor = sys.argv[sys.argv.index('--cnn')+1]
        kmertype = load_cnn_model(predictor, verbose = False)
        kk = '-1'
        outname += 'cnn'+str(kk)
    elif '--kmer' in sys.argv:
        kk = int(sys.argv[sys.argv.index('--kmer')+1])
        gap = int(sys.argv[sys.argv.index('--kmer')+2])
        kmertype = sys.argv[sys.argv.index('--kmer')+3]
        outname += 'kmer'+kmertype+str(kk)+'-'+str(gap)
    
    distance = None
    distance_embedding = False
    if '--distance_embedding' in sys.argv:
        distance_embedding = True
        distance = sys.argv[sys.argv.index('--distance_embedding')+1]
    
    embmethod = 'pca'
    params = {}
    if '--visual_method' in sys.argv:
        embmethod = sys.argv[sys.argv.index('--visual_method')+1]
        if len(sys.argv) > sys.argv.index('--visual_method')+2 and '--' not in sys.argv[sys.argv.index('--visual_method')+2]:
            embmethparms = sys.argv[sys.argv.index('--visual_method')+2]
            if '+' in embmethparms:
                embmethparms=embmethparms.split('+')
            else:
                embmethparms=[embmethparms]
            params = {}
            for e, embm in enumerate(embmethparms):
                if '=' in embm:
                    embm = embm.split('=')
                elif ':' in embm:
                    embm = embm.split(':')
                if embm[0] == 'distance':
                    distance = embm[1]
                else:
                    params[embm[0]] = check(embm[1])
            # check list of parameters that have to provided to tnse and umap
            # decide on distance measure, aka metric
            # decide on k-nearest n_neighbors
            # decide on other visualization parameters
            
    outname += embmethod
    
    
    embed = embedding(n_components = np.amax(components)+1, reglen = kk, center = center, distance = distance, distance_embedding = distance_embedding, regtype = regtype, representation = kmertype, gaplength = gap, algorithm = embmethod)
    
    orig_emb = embed.seq_rep(orig_seqs)
    opt_emb = embed.seq_rep(opt_seqs)
    
    if '--jointembedding' in sys.argv:
        outname += 'jointemb'
        trainembed = np.append(orig_emb, opt_emb, axis = 0)
    
    
    if '--seed_set' in sys.argv:
        outname += '_seeds'
        seed_names, seed_seqs = readinfasta(sys.argv[sys.argv.index('--seed_set')+1])
        seed_emb = embed.seq_rep(seed_seqs)
        if '--jointembedding' in sys.argv:
            trainembed = np.append(trainembed, seed_emb, axis = 0)
    
    if '--jointembedding' in sys.argv:
        orig_2d = embed.fit(trainembed, **params)
        opt_2d = orig_2d[len(orig_emb):len(orig_emb)+len(opt_emb)]
        if '--seed_set' in sys.argv:    
            seed_2d = orig_2d[len(orig_emb)+len(opt_emb):]
        orig_2d = orig_2d[:len(orig_emb)]
        trainembed = None
    else:
        orig_2d = embed.fit(orig_emb, **params)
        opt_2d = embed.transform(opt_emb)
        if '--seed_set' in sys.argv:
            seed_2d = embed.transform(seed_emb)
    
    orig_2d, opt_2d = orig_2d[:,components], opt_2d[:,components]
    if '--seed_set' in sys.argv:
        seed_2d = seed_2d[:, components]
    
    connections = None
    if '--seed_set' in sys.argv:
        connections = []
        if '--connect_seeds' in sys.argv:
            for o, opt_name in enumerate(opt_names):
                opt_name = opt_name.rsplit('_',1)[0]
                if opt_name in seed_names:
                    io = list(seed_names).index(opt_name)
                    connections.append([[opt_2d[o][0],seed_2d[io][0]],[opt_2d[o][1],seed_2d[io][1]]])
    
    
    if '--target_set' in sys.argv:
        target_set = np.genfromtxt(sys.argv[sys.argv.index('--target_set')+1], dtype = str)
        rand = np.isin(orig_names, target_set)
        target_seqs, target_emb, target_2d = orig_seqs[rand], orig_emb[rand], orig_2d[rand]
    
    

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.scatter(orig_2d[:,0], orig_2d[:,1], label = 'input', c = 'grey', alpha = 0.5)
    if '--seed_set' in sys.argv:
        ax.scatter(seed_2d[:,0], seed_2d[:,1], label = 'seeds', c = 'purple', alpha = 0.6)
    if '--target_set' in sys.argv:
        ax.scatter(target_2d[:,0], target_2d[:,1], label = 'target', c = 'k', alpha = 0.6)
    
    ax.scatter(opt_2d[:,0], opt_2d[:,1], label = 'optimized', c = 'limegreen')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if connections is not None:
        for con in connections:
            ax.plot(con[0], con[1], c = 'grey', lw = 0.5)
    ax.legend()
    if '--savefig' in sys.argv:
        fig.savefig(outname+'_seqdistribution.jpg', dpi = 200)
    else:
        plt.show()
    
    # could quantify this by plotting the minimum distance between real sequences, between optimized sequences, between seed sequences, and between optimized and real sequences. 

    if '--assess_distance' in sys.argv:
        distance_metric = sys.argv[sys.argv.index('--assess_distance')+1]
        
        if distance_metric == "edit" or distance_metric == 'Levenshtein' or distance_metric == 'identity' or distance_metric == 'Identity' or distance_metric == 'ID':
            #traintrain = distance_wrap(orig_seqs, distance =distance_metric)
            #trainopt = distance_wrap(opt_seqs, orig_seqs, distance =distance_metric)
            optopt = distance_wrap(opt_seqs, distance =distance_metric)
            if '--seed_set' in sys.argv:
                if '--target_set' in sys.argv:
                    seedtrain = distance_wrap(seed_seqs, target_seqs, distance =distance_metric)
                else:
                    seedtrain = distance_wrap(seed_seqs, orig_seqs, distance =distance_metric)
            if '--target_set' in sys.argv:
                opttarget = distance_wrap(opt_seqs, target_seqs, distance =distance_metric)
                targettarget = distance_wrap(target_seqs, target_seqs, distance =distance_metric)
        else:
            #traintrain = distance_wrap(orig_emb, distance =distance_metric)
            trainopt = distance_wrap(opt_emb, orig_emb, distance =distance_metric)
            optopt = distance_wrap(opt_emb, distance =distance_metric)
            if '--seed_set' in sys.argv:
                if '--target_set' in sys.argv:
                    seedtrain = distance_wrap(seed_emb, target_emb, distance =distance_metric)
                else:
                    seedtrain = distance_wrap(seed_emb, orig_emb, distance =distance_metric)
            if '--target_set' in sys.argv:
                opttarget = distance_wrap(opt_emb, target_emb, distance =distance_metric)
                targettarget = distance_wrap(target_emb, target_emb, distance =distance_metric)
                
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #ax.hist(traintrain, color = 'grey', alpha = 0.3, density = True)
        ax.hist(np.sort(targettarget, axis = 1)[:,1], color = 'grey', alpha = 0.3, density = True, label = 'between target')
        ax.hist(np.sort(optopt,axis = 1)[:,1], color = 'steelblue', alpha = 0.4, density = True, label = 'opt to opt')
        if '--seed_set' in sys.argv:
            ax.hist(np.amin(seedtrain, axis = 1), color = 'purple', alpha = 0.4, density = True, label = 'seed to target')
        if '--target_set' in sys.argv:
            ax.hist(np.amin(opttarget, axis = 1), color = 'limegreen', alpha = 0.8, density = True, label = 'opt to target')
        
        #ax.hist(np.amin(trainopt,axis = 1), color = 'green', alpha = 0.5, density = True)
        ax.set_xlabel(distance_metric)
        ax.legend()
        if '--savefig' in sys.argv:
            fig.savefig(outname+distance_metric+'_disthist.jpg', dpi = 200)
        else:
            plt.show()
        
        
        
            
                
                

    
