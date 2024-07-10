import numpy as np
import sys, os
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from scipy.spatial.distance import cdist
import umap
from scipy.linalg import svd
import sklearn.manifold as skm
from data_processing import check

def reduce_dim(X, red, var, center = True):
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
    
from functools import reduce


def read(outputfile, delimiter = None):
    if os.path.isfile(outputfile):
        if os.path.splitext(outputfile)[1] == '.npz':
            Yin = np.load(outputfile, allow_pickle = True)
            print(Yin.files)
            if 'counts' in Yin.files:
                Y = Yin['counts']
            elif 'values' in Yin.files:
                Y = Yin['values']
            outputnames = Yin['names'] # Y should of shape (nexamples, nclasses, l_seq/n_resolution)
        else:
            Yin = np.genfromtxt(outputfile, dtype = str, delimiter = delimiter)
            Y, outputnames = Yin[:, 1:].astype(float), Yin[:,0]
    elif ',' in outputfile:
        Y, outputnames = [], []
        for putfile in outputfile.split(','):
            if os.path.splitext(putfile)[1] == '.npz':
                Yin = np.load(putfile, allow_pickle = True)
                onames = Yin['names']
                sort = np.argsort(onames)
                if 'counts' in Yin.files:
                    yname = 'counts'
                elif 'values' in Yin.files:
                    yname = 'values'
                Y.append(Yin[yname][sort])
                outputnames.append(onames[sort])
            else:
                Yin = np.genfromtxt(putfile, dtype = str, delimiter = delimiter)
                onames = Yin[:,0]
                sort = np.argsort(onames)
                Y.append(Yin[:, 1:].astype(float)[sort]) 
                outputnames.append(onames[sort])
            
        comnames = reduce(np.intersect1d, outputnames)
        for i, yi in enumerate(Y):
            Y[i] = yi[np.isin(outputnames[i], comnames)]
        outputnames = comnames
        Y = np.concatenate(Y, axis = 1)
    u_, sort = np.unique(outputnames, return_index = True)
    outputnames, Y = outputnames[sort], Y[sort]
    print(len(outputnames), np.shape(Y))
    return outputnames, Y
    

def plot_2d(orig_2d, colors = None, cmap = None, alpha = 0.5, figsize = (3.5,3.5), size = 5, outname = None, dpi = 200, vlim = None, sortbycolor = 0):
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if vlim is None:
        if colors is not None:
            vmin, vmax = np.amin(colors), np.amax(colors)
            if vmin < 0:
                vabs = np.amax(abs(vmin), abs(vmax))
                vmin, vmax = -vabs, vabs
            print(vmin, vmax)
    if sortbycolor != 0:
        if isinstance(colors, np.ndarray):
            sortcolors = np.copy(colors)
            if abs(sortbycolor) > 1:
                sortcolors = np.absolute(colors)
            if sortbycolor < 0:
                sortcolors = -sortcolors
            sort = np.argsort(sortcolors)
            colors, orig_2d = colors[sort], orig_2d[sort]
            
    ap = ax.scatter(orig_2d[:,0], orig_2d[:,1], c = colors, alpha = alpha, cmap = cmap, s = size)
    fig.colorbar(ap, aspect = 2, pad = 0, anchor = (0,0.9), shrink = 0.15)
    if outname is not None:
        fig.savefig(outname+'_2d.jpg', dpi = dpi)
    else:
        plt.show()
    
    




if __name__ == '__main__':
    
    names, x_emb = read(sys.argv[1])
    
    outname = os.path.splitext(sys.argv[1])[0]
    if '--embedded' in sys.argv:
        xy = x_emb
    else:
        params = {"norm" : 'original', "normvar" : 1., "algorithm" : 'umap', "n_components" : 2}
        if '--embedding' in sys.argv:
            embmethparms = sys.argv[sys.argv.index('--embedding')+1]
            if '+' in embmethparms:
                embmethparms=embmethparms.split('+')
            else:
                embmethparms=[embmethparms]

            for e, embm in enumerate(embmethparms):
                if '=' in embm:
                    embm = embm.split('=')
                elif ':' in embm:
                    embm = embm.split(':')
                params[embm[0]] = check(embm[1])
                print(embm[0], check(embm[1]))
        for p, par in enumerate(params):
            outname += '_'+str(par)[:2] + str(params[par])[:3]
        
        print(params)
        embed = embedding(**params)
        xy = embed.fit(x_emb)
    
        if '--save_embedding' in sys.argv:
            np.savez_compressed(outname, names = names, values = xy)
    
    if '--addname' in sys.argv:
        outname += '_'+sys.argv[sys.argv.index('--addname') + 1]
    print(outname)
    
    #include option for various colormaps that can be mixed in one umap.
    viskwargs = {}
    if '--colors' in sys.argv:
        cfile = sys.argv[sys.argv.index('--colors')+1]
        ccont = int(sys.argv[sys.argv.index('--colors')+2])
        outname += 'col'+str(ccont)
        if os.path.splitext(cfile)[-1] == '.npz':
            cnames, colors = read(cfile)
            colors = colors[:, ccont]
            print(cnames, colors)
        else:
            cfile = np.genfromtxt(cfile, dtype = str)
            cnames, colors = cfile[:,0], cfile[:, ccont].astype(float)
        sort = np.argsort(cnames)[np.isin(np.sort(cnames),names)]
        cnames, colors = cnames[sort], colors[sort]
        if not np.array_equal(cnames,names):
            print('Colors dont match names in embedding')
            sys.exit()
        else:
            viskwargs['colors'] = colors
        if '--transform_color' in sys.argv:
            colors = 1. - colors
    
    if '--cmap' in sys.argv:
        cmap = sys.argv[sys.argv.index('--cmap')+1]
        if ',' in cmap:
            cmap = ListedColormap(cmap.split(','))
        viskwargs['cmap'] = cmap
    
    if '--plot_params' in sys.argv:
        visparms = sys.argv[sys.argv.index('--plot_params')+1]
        if '+' in visparms:
            visparms=visparms.split('+')
        else:
            visparms=[visparms]
        for v, vm in enumerate(visparms):
            if '=' in vm:
                vm = vm.split('=')
            elif ':' in embm:
                vm = vm.split(':')
            viskwargs[vm[0]] = check(vm[1])
    for p, par in enumerate(viskwargs):
        if not isinstance(viskwargs[par], np.ndarray) and not isinstance(viskwargs[par], list):
            outname += '_'+str(par)[:2] + str(viskwargs[par])[:3]
    
    if '--savefig' in sys.argv:
        viskwargs['outname'] = outname
        if '--outname' in sys.argv:
            viskwargs['outname'] = sys.argv[sys.argv.index('outname')+1]
            
    plot_2d(xy, **viskwargs)

    
        
