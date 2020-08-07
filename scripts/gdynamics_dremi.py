import pandas as pd
import os
import glob
import pickle
import phate
import scprep
import meld
import graphtools as gt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import datetime
import scanpy as sc
from sklearn.decomposition import PCA
from scipy import sparse
import random



# settings
plt.rc('font', size = 9)
plt.rc('font', family='sans serif')
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42
plt.rcParams['text.usetex']=False
plt.rcParams['legend.frameon']=False
plt.rcParams['axes.grid']=False
plt.rcParams['legend.markerscale']=0.5
sc.set_figure_params(dpi=300,dpi_save=600,
                     frameon=False,
                     fontsize=9)
plt.rcParams['savefig.dpi']=600
sc.settings.verbosity=2
sc._settings.ScanpyConfig.n_jobs=-1
sns.set_style("ticks")


# fps
dfp = '/home/ngr4/project/sccovid/data/'
pfp = '/home/ngr4/project/sccovid/results/'
pdfp = '/home/ngr4/project/sccovid/data/processed/'
sc.settings.figdir = pfp


def loader(fname,fpath,backed=None) : 
    start = time.time()
    adata = sc.read_h5ad(filename=os.path.join(fpath,fname),backed=backed)
    print('loaded @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))
    print('took {:.2f}-s to load data'.format(time.time()-start))
    return adata

def writer(fname,fpath,AnnData) :
    start = time.time()
    AnnData.write(os.path.join(fpath,fname))
    print('saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))
    print('took {:.2f}-s to save data'.format(time.time()-start))
    

if True :
    # load personal
    fname='scv2_200428.h5ad'
    adata = loader(fname,pdfp)

    
# meld
adata.obs['res_t'] = adata.obs['Condition'].astype(str)
adata.obs['res_t'][adata.obs['Condition']=='Mock']=0
adata.obs['res_t'][adata.obs['Condition']=='1dpi']=1
adata.obs['res_t'][adata.obs['Condition']=='2dpi']=2
adata.obs['res_t'][adata.obs['Condition']=='3dpi']=3

G = gt.Graph(data=adata.uns['neighbors']['connectivities']+sparse.diags([1]*adata.shape[0],format='csr'),
                 precomputed='adjacency',
                 use_pygsp=True)
G.knn_max = None
adata.obs['ees_t']=meld.MELD().fit_transform(G=G,RES=adata.obs['res_t'].to_numpy(dtype=float))
adata.obs['ees_t']=adata.obs['ees_t']-adata.obs['ees_t'].mean() # mean center

del G

# cluster genes
random_genes = False

if random_genes:
    genes = adata.var_names.to_list()
    genes = random.sample(random_genes, 10)
else:
    genes = adata.var_names.to_list()
#     genes=[int(sys.argv[1]:int(sys.argv[2]))]
print('Aggregating data')
gdata = pd.DataFrame()
Y = pd.DataFrame()
Y['Condition'] = adata.obs['Condition'].to_list()
Y['Infected'] = adata.obs['scv2_10+'].map({1:'Infected',0:'Bystander'}).to_list()
Y.loc[Y['Condition']=='Mock','Infected'] = 'Mock'
Y['Inferred time'] = adata.obs['ees_t'].to_list()  
imputed_data = pd.DataFrame(adata.layers['imputed_bbknn'], columns=adata.var_names)
del adata
tic = time.time()

for j,gene in enumerate(genes): 
    if j % 100 == 0:
        iter_left = len(genes) - (j+1)
        p_left=100*(j+1)/len(genes)
        toc = time.time()-tic
        print('  data aggregated for {:.1f}-% genes\tin {:.2f}-s\t~{:.2f}-min remain'.format(p_left,toc,((toc/(j+1))*iter_left)/60))
    X = Y
    X[gene] = imputed_data[gene]


    # DREMI-plots
    nbins = 20
    norm = True
    x=X.loc[(X['Infected']=='Infected') | (X['Infected']=='Mock'), 'Inferred time']
    y=X.loc[(X['Infected']=='Infected') | (X['Infected']=='Mock'), gene]
    H, x_edges, y_edges = np.histogram2d(x, y, 
                                     bins=nbins, density=False,
                                     range=[[np.quantile(x, q=0.0275),
                                             np.quantile(x, q=0.975)],
                                            [0,
                                             np.quantile(y, q=0.99)]])
    if norm:
        H = H / H.sum(axis=0)
        H[np.isnan(H)] = 0

    inf = np.reshape(H,-1)
    
    del H

    x=X.loc[(X['Infected']=='Bystander') | (X['Infected']=='Mock'), 'Inferred time']
    y=X.loc[(X['Infected']=='Bystander') | (X['Infected']=='Mock'), gene]
    H, x_edges, y_edges = np.histogram2d(x, y, 
                                     bins=nbins, density=False,
                                     range=[[np.quantile(x, q=0.0275),
                                             np.quantile(x, q=0.975)],
                                            [0,
                                             np.quantile(y, q=0.99)]])
    if norm:
        H = H / H.sum(axis=0)
        H[np.isnan(H)] = 0

    uninf = np.reshape(H,-1)
    
    dt = pd.DataFrame(np.append(inf,uninf))
    dt = dt.T
    dt['gene'] = gene

    gdata = gdata.append(dt, ignore_index=False)
    
    del X, dt, H, inf, uninf

if True: 
    # save
    gdata.to_csv(os.path.join(pdfp,'gdynamics_concat.csv'))
