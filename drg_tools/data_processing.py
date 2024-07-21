# data_processing.py

'''
Contains functions to modify the content of data matrices

'''

import sys, os 
import numpy as np



def reduce_dim(X, red, var, center = True):
    '''
    Reduces the dimension of a matrix of size N_data, d_features
    '''
    if center:
        Xmean = np.mean(X, axis = 0)
        X = X - Xmean
    if 'standard' in red:
        X /= np.std(X, axis = 0)
    U, S, Vh = svd(X, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd')
    print(np.shape(U))
    # check if singular values are in correct order
    if not np.array_equal(np.argsort(-S), np.arange(len(S))):
        print('Singular values not sorted')
    if var > 1:
        U, S, Vh = U[:,:var], S[:var], Vh[:var]
    else: 
        Sv = S**2
        SV = np.cumsum(Sv)/np.sum(Sv)
        var = np.where(SV < var)[0][-1] + 1
        print('Selected', var, 'singular values')
        U, S, Vh = U[:,:var], S[:var], Vh[:var]
        print(SV[:var])
    if 'normed' not in red:
        U = U * S
    return U


class embedding():
    def __init__(self, norm = 'standard', normvar = 0.9, algorithm = 'umap', n_components = 2, metric = 'euclidean', **kwargs):
        self.normvar = normvar
        self.algorithm = algorithm
        self.center = True
        self.norm = norm
        if self.algorithm == 'NMF':
            self.center = False
            self.norm = 'none'
        self.mean = None
        self.std = None
        self.xfit = None
        self.n_components = n_components
        self.metric = 'euclidean'
        self.kwargs = kwargs
        
    def fit(self,x):
        print(self.normvar, self.algorithm)
        if self.normvar != 1:
            x = reduce_dim(x, self.norm, self.normvar, center = self.center)
        
        if self.algorithm == 'svd' or self.algorithm == 'SVD':
            self.decoder = TruncatedSVD(n_components=self.n_components, n_iter=7, random_state=42)
        elif self.algorithm == 'NMF':
            self.decoder = NMF(n_components=self.n_components, init = 'nndsvda')
        elif self.algorithm == 'pca' or self.algorithm == 'PCA':
            self.decoder = PCA(n_components=self.n_components)
        elif self.algorithm == 'TSNE' or self.algorithm == 'tsne':
            print('Fit TSNE')
            self.decoder = skm.TSNE(n_components=self.n_components, metric = self.metric, **self.kwargs)
            # perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', metric_params=None, init='warn', verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='deprecated'
        elif self.algorithm == 'umap' or self.algorithm == 'UMAP':
            print('Fit UMAP')
            self.decoder = umap.UMAP(n_components=self.n_components, metric = self.metric, **self.kwargs)
            # n_neighbors=15, n_components=2, metric='euclidean', metric_kwds=None, output_metric='euclidean', output_metric_kwds=None, n_epochs=None, learning_rate=1.0, init='spectral', min_dist=0.1, spread=1.0, low_memory=True, n_jobs=-1, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0, negative_sample_rate=5, transform_queue_size=4.0, a=None, b=None, random_state=None, angular_rp_forest=False, target_n_neighbors=-1, target_metric='categorical', target_metric_kwds=None, target_weight=0.5, transform_seed=42, transform_mode='embedding', force_approximation_algorithm=False, verbose=False, tqdm_kwds=None, unique=False, densmap=False, dens_lambda=2.0, dens_frac=0.3, dens_var_shift=0.1, output_dens=False, disconnection_distance=None, precomputed_knn=(None, None, None)
        if self.algorithm == 'TSNE' or self.algorithm == 'tsne':
            return self.decoder.fit_transform(x)
        
        self.decoder.fit(x)
        print('Done fit')
        return self.transform(x)
    
    def transform(self,x):
        return self.decoder.transform(x)





def manipulate_input(X, features, sysargv):
    if '--select_features' in sysargv:
        selected_feat = np.genfromtxt(sysargv[sysargv.index('--select_features')+1], dtype = str)
        if len(features[0]) == len(selected_feat[0]):
            featmask = np.isin(features, selected_feat)
        elif len(features[0]) < len(selected_feat[0]):
            selected_feat = ','.join(selected_feat)
            s_feat = []
            for feat in feature:
                if feat in selected_feat:
                    s_feat.append(feat)
            featmask = np.isin(features, s_feat)
        elif len(features[0]) > len(select_features[0]):
            selected_feat = ','.join(selected_feat)
            s_feat = []
            for feat in feature:
                if feat in selected_feat:
                    s_feat.append(feat)
            featmask = np.isin(features, s_feat)
        features, X = np.array(features)[featmask], X[:, featmask]
        print('X reduced to', np.shape(X))
        outname+= '_featsel'+str(len(X[0]))
        
        
    if '--centerfeature' in sysargv:
        outname += '-cenfeat'
        X = X - np.mean(X, axis = 0)
    
    elif '--centerdata' in sysargv:
        outname += '-cendata'
        X = X - np.mean(X, axis = 1)[:, None]


    if '--norm2feature' in sysargv:
        outname += '-n2feat'
        norm = np.sqrt(np.sum(X*X, axis = 0))
        norm[norm == 0] = 1.
        X = X/norm
        
    elif '--norm1feature' in sysargv:
        outname += '-n1feat'
        norm =np.sum(np.absolute(X), axis = 0)
        norm[norm == 0] = 1.
        X = X/norm
    
    if '--norm2data' in sysargv:
        outname += '-n2data'
        norm =np.sqrt(np.sum(X*X, axis = 1))[:, None] 
        X = X/norm
        
    elif '--norm1data' in sysargv:
        outname += '-n1data'
        X = X/np.sum(np.absolute(X), axis = 1)[:,None]
        
    return X, features


