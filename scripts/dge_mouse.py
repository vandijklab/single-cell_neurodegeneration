'''
data loader. Also, cluster plots and DGE per timepoint per cluster files are saved

Data can be found
[here](http://fcb.ycga.yale.edu:3010/Cow4QziaOJKZZv7NnIWBF1BqC653y/011520/),
[here](http://fcb.ycga.yale.edu:3010/AbZmfWylm7iuPPfaizz183RvaUyp3/121819),
and [here](http://fcb.ycga.yale.edu:3010/1lUhZNv4dlTxjT6c7ezxKqjnNVGwy/012320)

Filepath on cluster: `/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/*`

The animal identities are as follows:

- 5wk WT= 7202, 72921, 72922
- 5wk SCA1= 7294, 72931, 72932
- 12wk wild-type: 22018, 2061, 2062
- 12wk SCA1: 22019, 2063, 2065
- 18wk WT: 6569, 65701, 65702
- 18wk SCA1: 6571, 65731, 65732
- 24wk wild-type: 1974, 2020, 20202
- 24wk SCA1: 1589, 2021, 20212
- 30wk WT: 5812, 58231, 58232
- 30wk SCA1: 5822, 58241, 58242

Neal G. Ravindra, 11 Feb 2020

NOTES: if running with MAGIC, need ~200G per cpu (600G across 3CPUs worked)

'''

import os, sys, glob, re, math, pickle
import phate,scprep,magic,meld
import graphtools as gt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time,random,datetime
import networkx as nx
import scvelo as scv
from sklearn import metrics
from sklearn import model_selection
from scipy import sparse
from scipy.stats import mannwhitneyu, tiecorrect, rankdata
from statsmodels.stats.multitest import multipletests
import scanpy as sc
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import SpectralClustering, OPTICS, cluster_optics_dbscan, AgglomerativeClustering

# settings
plt.rc('font', size = 8)
plt.rc('font', family='sans serif')
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42
plt.rcParams['text.usetex']=True
plt.rcParams['legend.frameon']=False
plt.rcParams['axes.grid']=False
plt.rcParams['legend.markerscale']=0.5
sc.set_figure_params(dpi=300,dpi_save=600,
                     frameon=False,
                     fontsize=8)
plt.rcParams['savefig.dpi']=600
sc.settings.verbosity=2
sc._settings.ScanpyConfig.n_jobs=-1



# reproducibility
rs = np.random.seed(42)

# utils
def mwu(X,Y,gene_names,correction=None,debug=False) :
    '''
    Benjamini-Hochberg correction implemented. Can change to Bonferonni

    gene_names (list)
    if X,Y single gene expression array, input x.reshape(-1,1), y.reshape(-1,1)

    NOTE: get zeros sometimes because difference (p-value is so small)
    '''
    p=pd.DataFrame()
    print('Mann-Whitney U w/Benjamini/Hochberg correction\n')
    start = time.time()
    for i,g in enumerate(gene_names) :
        if i==np.round(np.quantile(np.arange(len(gene_names)),0.25)) :
            print('... 25% completed in {:.2f}-s'.format(time.time()-start))
        elif i==np.round(np.quantile(np.arange(len(gene_names)),0.5)) :
            print('... 50% completed in {:.2f}-s'.format(time.time()-start))
        elif i==np.round(np.quantile(np.arange(len(gene_names)),0.75)) :
            print('... 75% completed in {:.2f}-s'.format(time.time()-start))
        p.loc[i,'Gene']=g
        if (tiecorrect(rankdata(np.concatenate((np.asarray(X[:,i]),np.asarray(Y[:,i])))))==0) :
            if debug :
                print('P-value not calculable for {}'.format(g))
            p.loc[i,'pval']=np.nan
        else :
            _,p.loc[i,'pval']=mannwhitneyu(X[:,i],Y[:,i]) # continuity correction is True
    print('... mwu computed in {:.2f}-s\n'.format(time.time() - start))
    # ignore NaNs, since can't do a comparison on these (change numbers for correction)
    p_corrected = p.loc[p['pval'].notna(),:]
    new_pvals = multipletests(p_corrected['pval'],method='fdr_bh')
    p_corrected['pval_corrected'] = new_pvals[1]
    return p_corrected

def log2aveFC(X,Y,gene_names,AnnData=None) :
    '''not sensitivity to directionality due to subtraction

    X and Y full arrays, subsetting performed here

    `gene_names` (list): reduced list of genes to calc

    `adata` (sc.AnnData): to calculate reduced list. NOTE: assumes X,Y drawn from adata.var_names
    '''
    if not AnnData is None :
        g_idx = [i for i,g in enumerate(AnnData.var_names) if g in gene_names]
        fc=pd.DataFrame({'Gene':AnnData.var_names[g_idx],
                         'log2FC':np.log2(X[:,g_idx].mean(axis=0)) - np.log2(Y[:,g_idx].mean(axis=0))}) # returns NaN if negative value 
    else :
        fc=pd.DataFrame({'Gene':gene_names,
                         'log2FC':np.log2(X.mean(axis=0)) - np.log2(Y.mean(axis=0))})
    return fc

# fps
dfp = '/home/ngr4/project/scnd/data/'
pfp = '/home/ngr4/project/scnd/results/'
pdfp = '/home/ngr4/project/scnd/data/processed/'
sc.settings.figdir = pfp

# loader
#                 - 5wk WT= 7202, 72921, 72922
#                 - 5wk SCA1= 7294, 72931, 72932
data_folders = ['/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121819/7202_MNT_cellranger/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121819/72921_MNT_cellranger/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121819/72922_MNT_cellranger/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121819/7294_MNT_cellranger/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121819/72931_MNT_cellranger/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121819/72932_MNT_cellranger/filtered_feature_bc_matrix/',
#                 - 12wk wild-type: 22018, 2061, 2062
#                 - 12wk SCA1: 22019, 2063, 2065
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/011520/22018CB_MNT/outs/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/011520/2061CB_MNT/outs/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/011520/2062CB_MNT/outs/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/011520/22019CB_MNT/outs/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/011520/2063CB_MNT/outs/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/011520/2065CB_MNT/filtered_feature_bc_matrix/',
#                 - 18wk WT: 6569, 65701, 65702
#                 - 18wk SCA1: 6571, 65731, 65732
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121819/6569_MNT_cellranger/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121819/65701_MNT_cellranger/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121819/65702_MNT_cellranger/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121819/6571_MNT_cellranger/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121819/65731_MNT_cellranger/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121819/65732_MNT_cellranger/filtered_feature_bc_matrix/',

#                 - 24wk wild-type: 1974, 2020, 20202
#                 - 24wk SCA1: 1589, 2021, 20212
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/011520/1974CB_MNT/outs/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/011520/2020CB_MNT/outs/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/012320/20202CB_MNT/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/011520/1589CB_MNT/outs/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/011520/2021CB_MNT/outs/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/011520/20212CB_MNT/outs/filtered_feature_bc_matrix/',
#                 - 30wk WT: 5812, 58231, 58232
#                 - 30wk SCA1: 5822, 58241, 58242
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121919/5812_MNT/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121919/58231_MNT/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121919/58232_MNT/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121919/5822_MNT/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121919/58241_MNT/filtered_feature_bc_matrix/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/121919/58242_MNT/filtered_feature_bc_matrix/']

files_not_found = []
for i in data_folders :
    if not os.path.exists(i) :
        files_not_found.append(i)
    if not files_not_found == [] :
        print('Folders not found...')
        for j in files_not_found :
            print(j)
        raise IOError('Change path to data')

total = time.time()

if False :
    # first load
    running_cellcount=0
    start = time.time()
    adatas = {}
    for i,folder in enumerate(data_folders) :
        animal_id = os.path.split(re.findall('(.*)_MNT',folder)[0])[1]
        if 'CB' in animal_id :
            animal_id = animal_id[:-2]
        print('... storing %s into dict (%d/%d)' % (animal_id,i+1,len(data_folders)))
        adatas[animal_id] = sc.read_10x_mtx(folder)
        running_cellcount+=adatas[animal_id].shape[0]
        print('...     read {} cells; total: {} in {:.2f}-s'.format(adatas[animal_id].shape[0],running_cellcount,time.time()-start))
    batch_names = list(adatas.keys())
    print('\n... concatenating of {}-samples'.format(len(data_folders)))
    adata = adatas[batch_names[0]].concatenate(adatas[batch_names[1]],adatas[batch_names[2]],
                                               adatas[batch_names[3]],adatas[batch_names[4]],
                                               adatas[batch_names[5]],adatas[batch_names[6]],
                                               adatas[batch_names[7]],adatas[batch_names[8]],
                                               adatas[batch_names[9]],adatas[batch_names[10]],
                                               adatas[batch_names[11]],adatas[batch_names[12]],
                                               adatas[batch_names[13]],adatas[batch_names[14]],
                                               adatas[batch_names[15]],adatas[batch_names[16]],
                                               adatas[batch_names[17]],adatas[batch_names[18]],
                                               adatas[batch_names[19]],adatas[batch_names[20]],
                                               adatas[batch_names[21]],adatas[batch_names[22]],
                                               adatas[batch_names[23]],adatas[batch_names[24]],
                                               adatas[batch_names[25]],adatas[batch_names[26]],
                                               adatas[batch_names[27]],adatas[batch_names[28]],
                                               adatas[batch_names[29]],
                                               batch_categories = batch_names)
    print('Raw load in {:.2f}-min'.format((time.time() - start)/60))

    if True :
        # save
        adata.write(os.path.join(pdfp,'mouse_MT_bbknn.h5ad'))
        print('\n... saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))

    if True :
        # annotate metadata
        WT = ['7202', '72921', '72922',   #5wk
              '22018', '2061', '2062',    #12wk
              '6569', '65701', '65702',
              '1974', '2020', '20202',    #24wk
              '5812', '58231', '58232']   #30wk
        SCA1 = ['7294', '72931', '72932', #5wk
                '22019', '2063', '2065',  #12wk
                '6571', '65731', '65732',
                '1589', '2021', '20212',  #24wk
                '5822', '58241', '58242'] #30wk
        wk5 = ['7202', '72921', '72922',  #WT
               '7294', '72931', '72932']  #SCA1
        wk12 = ['22018', '2061', '2062',
                '22019', '2063', '2065']
        wk18 = ['6569', '65701', '65702',
                '6571', '65731', '65732']
        wk24 = ['1974', '2020', '20202',
                '1589', '2021', '20212']
        wk30 = ['5812', '58231', '58232',
                '5822', '58241', '58242']
        genotype = []
        for i in adata.obs['batch'] : # verbose loop for quality-assurance
            if i in WT :
                genotype.append('WT')
            elif i in SCA1 :
                genotype.append('SCA1')
            else :
                raise ValueError('Encountered unclassifiable genotype for animal {}'.format(i))
        adata.obs['genotype']=genotype
        timepoint = []
        for i in adata.obs['batch'] :
            if i in wk5 :
                timepoint.append('5wk')
            elif i in wk12 :
                timepoint.append('12wk')
            elif i in wk18 :
                timepoint.append('18wk')
            elif i in wk24 :
                timepoint.append('24wk')
            elif i in wk30 :
                timepoint.append('30wk')
            else :
                raise ValueError('Encountered unclassifiable timepoint for animal {}'.format(i))
        adata.obs['timepoint']=timepoint

    if True :
        print(adata)

    # filter cells/genes, transform
    sc.pp.calculate_qc_metrics(adata,inplace=True)
    mito_genes = adata.var_names.str.startswith('mt-')
    adata.obs['pmito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
    print('Ncells=%d have >10percent mt expression' % np.sum(adata.obs['pmito']>0.1))
    print('Ncells=%d have <200 genes expressed' % np.sum(adata.obs['n_genes_by_counts']<200))
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3) # filtering cells gets rid of some genes of interest
    adata = adata[adata.obs.pmito <= 0.1, :]
    adata.raw = adata
    sc.pp.normalize_total(adata)
    sc.pp.sqrt(adata,chunked=True,chunk_size=10000)

    if True :
        # save
        adata.write(os.path.join(pdfp,'mouse_MT_bbknn.h5ad'))
        print('\n... saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))
        
    if True :
        # batch effect plot
        tdata = adata
        sc.tl.pca(tdata,n_comps=100)
        sc.pp.neighbors(tdata,n_neighbors=100,n_pcs=100)
        sc.tl.umap(tdata)
        fig,axarr=plt.subplots(5,2,figsize=(7,11))
        start = time.time()
        print('through')
        for i,t in enumerate(tdata.obs.timepoint.unique()) :
            pal18=['#ee5264','#565656','#75a3b7','#ffe79e','#fac18a','#f1815f','#ac5861','#62354f','#2d284b','#f4b9b9','#c4bbaf',
               '#f9ebae','#aecef9','#aeb7f9','#f9aeae','#9c9583','#88bb92','#bde4a7','#d6e5e3']
            pal8=sns.color_palette('colorblind',8)

            batches = tdata.obs.batch[tdata.obs.timepoint==t].unique()
            batch_inloop = tdata.obs.batch.astype(str)
            batch_inloop = ['Not {}'.format(t) if i not in batches else i for i in tdata.obs.batch]

            cmap_batch = {v:(pal8[i] if v in batches else 'gray') for i,v in enumerate(np.unique(batch_inloop))}

            scprep.plot.scatter2d(tdata.obsm['X_pca'],
                                  c=batch_inloop,
                                  cmap=cmap_batch,
                                  ticks=None,
                                  label_prefix='PCA',
                                  legend=False,
                                  ax=axarr[i,0],
                                  s = 0.05,
                                  alpha=0.6,
                                  rasterized=True,
                                  title='{} samples'.format(t))
            scprep.plot.scatter2d(tdata.obsm['X_umap'],
                                  c=batch_inloop,
                                  cmap=cmap_batch,
                                  ticks=None,
                                  label_prefix='UMAP',
                                  legend=True,
                                  legend_loc=(1.01,0.0),
                                  ax=axarr[i,1],
                                  s = 0.05,
                                  alpha=0.6,
                                  rasterized=True,
                                  title='{} samples'.format(t))

            print('... {} plot in {:.2f}-s'.format(i+1,time.time()-start))
        fig.savefig(os.path.join(pfp,'batchEffect_mouse_MT_bbknn.pdf'),dpi=300,bbox_inches='tight')
        del tdata

    
    # calc embeddings for batch corrected
    start = time.time()
    print('starting embeddings...')
    sc.tl.pca(adata,n_comps=100)
    sc.external.pp.bbknn(adata,batch_key='batch')
#     sc.pp.neighbors(adata, n_neighbors=100, n_pcs=100)
    sc.tl.louvain(adata,resolution=3)
    sc.tl.umap(adata)
    if True :
        # save adata obj with batch correction
        adata.write(os.path.join(pdfp,'mouse_MT_bbknn.h5ad'))
        print('\n... saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))
    print('... sc embeddings in {:.2f}-min'.format((time.time()-start)/60))

    # compute PHATE
    G = gt.Graph(data=adata.uns['neighbors']['connectivities']+sparse.diags([1]*adata.shape[0],format='csr'),
                 precomputed='adjacency',
                 use_pygsp=True)
    G.knn_max = None
    
    phate_op = phate.PHATE(knn_dist='precomputed',
                           gamma=0,
                           n_jobs=-1,
                           random_state=rs)
    adata.obsm['X_phate']=phate_op.fit_transform(G.K)

    if True :
        # save adata obj with batch correction
        adata.write(os.path.join(pdfp,'mouse_MT_bbknn.h5ad'))
        print('\n... saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))
    print('... full PHATE in {:.2f}-min'.format((time.time() - start)/60))


    if True :
        # MELD
        adata.obs['res_sca1']=[1 if i=='SCA1' else -1 for i in adata.obs['genotype']]
        adata.obs['ees_sca1']=meld.MELD().fit_transform(G=G,RES=adata.obs['res_sca1'])
        adata.obs['ees_sca1']=adata.obs['ees_sca1']-adata.obs['ees_sca1'].mean() # mean center
        if True :
            # save adata obj with batch correction
            adata.write(os.path.join(pdfp,'mouse_MT_bbknn.h5ad'))
            print('\n... saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))
            
    if True :
        # MAGIC
        magic_op=magic.MAGIC().fit(X=adata.X,graph=G) # running fit_transform produces wrong shape
        adata.layers['imputed_bbknn']=magic_op.transform(adata.X,genes='all_genes')
#         adata.layers['imputed_bbknn']=sparse.csr_matrix(magic_op.transform(adata.X,genes='all_genes')) # causes memory spike
        
        if True :
            # save adata obj with batch correction & imputation
            adata.write(os.path.join(pdfp,'mouse_MT_bbknn.h5ad'))
            print('\n... saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))

    print('Pre-processing dataset took {:.2f}-min'.format((time.time() - total)/60))

elif False :
    # save data objects
    start=time.time()
    adata.write(os.path.join(pdfp,'mouse_MT_bbknn.h5ad'))
    print('saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))
    print('... in {:.2f}-s'.format(time.time()-start))

else :
    # load data objects
    start=time.time()
    adata = sc.read_h5ad(os.path.join(pdfp,'mouse_MT_bbknn.h5ad'))
    print('loaded @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))
    print('... in {:.2f}-s'.format(time.time()-start))

        
    
print('\n----\n')

print('Starting clustering...')

if False :
    # markers
    finalmarkers = {'Granule cell':['Gabra6','Gm2694'],
                    'Excitatory neuron':['Slc17a6'],
                    'Purkinje cell':['Atp2a3','Calb1','Car8','Ppp1r17','Slc1a6'],
                    'Stellate/Basket/Golgi':['Megf10','Ntn1'],
                    'Astrocyte':['Aldh1l1','Aqp4','Slc1a3'],
                    'Bergmann glia':['Gdf10','Hopx','Timp4'],
                    'OL lineage':['Olig1','Olig2'],
                    'OPC':['C1ql1','Pdgfra','Ninj2'],
                    'OL':['Hapln2','Mag','Mog','Opalin'],
                    'Microglia':['C1qb','Cx3cr1','Dock2','P2ry12'],
                    'Pericytes':['Flt1','Pdgfrb','Rgs5'],
                    'Vascular':['Dcn','Lum'],
                    'Choroid plexus':['Folr1','Ttr']} 

    cluster = 'louvain'
    sc.pl.stacked_violin(adata,finalmarkers,
                         layer='imputed_bbknn',
                         use_raw=False,
                         groupby=cluster,
                         standard_scale='var',
                         var_group_rotation=90,
                         save = '_mouse_MT_bbknn.pdf')
    sc.pl.dotplot(adata,finalmarkers,
                  use_raw=True,
                  groupby=cluster,
                  standard_scale='var',
                  var_group_rotation=0.0,
                  color_map='Blues',
                  save = '_mouse_MT_bbknn.pdf')
    if False :
        fig,ax=plt.subplots(1,1,figsize=(6,8))
        scprep.plot.marker_plot(adata.X,adata.obs[cluster], # imputed here?
                                markers=finalmarkers, gene_names=adata.var_names,
                                reorder_markers=False,reorder_tissues=False,
                                cmap='bwr',
                                ax = ax)
        fig.savefig(os.path.join(pfp,'mouse_MT_bbknn_scprepdot.pdf'),bbox_inches='tight')
        sc.pl.heatmap(adata,finalmarkers,
                      use_raw=False,
                      groupby=cluster,
                      var_group_rotation=0.0,
                      standard_scale='var',
                      cmap='magma',
                      save = '_mouse_MT_bbknn.pdf')
    sc.pl.tracksplot(adata,finalmarkers,
                      use_raw=False,
                      groupby=cluster,
                      var_group_rotation=90,
                      standard_scale='var',
                      cmap='bwr',
                      save = '_mouse_MT_bbknn.png')

    if False :
        # large plot, requires imputed
        fig = plt.figure(figsize=(14,12))
        i = 0
        for k,v in markers.items() :
            for g in v :
                i+=1
                ax = fig.add_subplot(6,7,i)
                scprep.plot.scatter2d(adata.obsm['X_umap'],
                                      c = adata[:,g].layers['imputed'],
                                      cmap = 'magma',
                                      ticks = None,
                                      title = g,
                                      legend=False,
                                      s = 0.2,
                                      ax = ax)
        fig.savefig(os.path.join(pfp,'mouse_MT_bbknn_markersUMAP.png'),dpi=600, bbox_inches='tight')

if False :
    # viz what drives differences between clusters
    cluster = 'louvain'
    if False : # not a tested solution to dendogram recalc, mayb e in args of rank_genes_groups_plot
        print('Recalculating dendogram')
        sc.tl.dendrogram(adata,groupby=cluster)
    sc.tl.rank_genes_groups(adata, groupby=cluster,method='wilcoxon',n_genes=500)
    sc.plotting.rank_genes_groups_matrixplot(adata,n_genes=10,cmap='magma',standard_scale='var',save = '_mouse_MT_bbknn.png')
    sc.pl.rank_genes_groups(adata,n_genes=50,fontsize=6,save='_mouse_MT_bbknn.pdf')


    if True :
        # save adata obj with batch correction
        adata.write(os.path.join(pdfp,'mouse_MT_bbknn.h5ad'))
        print('\n... saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))

    if False :
        dt=sc.tl.marker_gene_overlap(adata, markers, method='overlap_coef', inplace=False)
        dt.to_csv(os.path.join(pfp,'mouse_MT_bbknn_overlapTop500.csv'))
        del dt
    
print('\nFinished loading, pre-processing, and clustering in {:.2f}-min'.format((time.time() - total)/60))


print('\n----\n')

print('Starting DGE per timepoint per cluster...')

if False :
    # color can get out of bounds so update range of colors so new one is called
    from matplotlib.colors import to_hex
    pal8=[to_hex(i,keep_alpha=True) for i in sns.cubehelix_palette(8)]
    adata.uns['louvain_colors']=np.array(['#ee5264','#565656','#75a3b7',
                                              '#ffe79e','#fac18a','#f1815f',
                                              '#ac5861','#62354f','#2d284b',
                                              '#f4b9b9','#c4bbaf','#f9ebae','#aecef9',
                                              '#aeb7f9','#f9aeae','#9c9583','#88bb92',
                                            '#bde4a7','#d6e5e3']+pal8)

if True :
    dge_grandtotal = time.time()
    cluster='ctype'
    fname = 'mouse_MT_imp' 
    for t in ['5wk','12wk','18wk','24wk','30wk'] :
        print('Evaluating {}'.format(t))
        t_total = time.time()
        dge_total = time.time()
        # load data obj
        tdata=sc.AnnData(adata[(adata.obs.timepoint==t),:]) # force not view

        # up down dichotomy
        print('\n--------')
        print('WT up...')
        print('--------\n')
        dge = pd.DataFrame()
        for i in tdata.obs[cluster].unique() :
            start = time.time()
            print('\n{}, WT vs SCA1'.format(i))
            print('----')
            X = tdata.layers['imputed_bbknn'][(tdata.obs[cluster]==i) & (tdata.obs['genotype']=='WT') ,:]
            Y = tdata.layers['imputed_bbknn'][(tdata.obs[cluster]==i) & (tdata.obs['genotype']=='SCA1') ,:]
            dt = scprep.stats.differential_expression(X,Y,
                                                       measure = 'emd',
                                                       direction='up',
                                                       gene_names=tdata.var_names,
                                                       n_jobs=-1)
            dt = pd.DataFrame({'Cell type':[i]*dt.shape[0],'Gene':dt.index,'EMD':dt['emd']})
            # mann-whitney u, corrected p-values
            p = mwu(X,Y,tdata.var_names)
            dt = pd.merge(dt,p,how='left',on="Gene")
            dt = dt.loc[dt['EMD']>0,:] # take only 'up' (switch for down)
            if np.sum(dt['pval_corrected']<0.01)<500:
                dt = dt.iloc[0:500,:]
                fc = log2aveFC(X,Y,dt['Gene'].to_list(),AnnData=tdata)
                dt = pd.merge(dt,fc,how='left',on='Gene')
            else :
                dt = dt.loc[dt['pval_corrected']<0.01,:] # may produce long rows
                fc = log2aveFC(X,Y,dt['Gene'].to_list(),AnnData=tdata) # flip X/Y for down
                dt = pd.merge(dt,fc,how='left',on='Gene')
            dge = dge.append(dt, ignore_index=True)
            print(dt.iloc[0:10,:])
            print('... computed in {:.2f}-s'.format(time.time()-start))
        dge.to_csv(os.path.join(pfp,'dge_'+t+'_'+fname+'_WTup.csv'),index=False)
        print('---- WT up in {:.2f}-min'.format((time.time() - dge_total)/60))

        if True :
            print('\n----------')
            print('WT down...')
            print('----------\n')
            dge_total = time.time()
            dge = pd.DataFrame()
            for i in tdata.obs[cluster].unique() :
                start = time.time()
                print('\n{}, WT vs SCA1'.format(i))
                print('----')
                X = tdata.layers['imputed_bbknn'][(tdata.obs[cluster]==i) & (tdata.obs['genotype']=='WT') ,:]
                Y = tdata.layers['imputed_bbknn'][(tdata.obs[cluster]==i) & (tdata.obs['genotype']=='SCA1') ,:]
                dt = scprep.stats.differential_expression(X,Y,
                                                           measure = 'emd',
                                                           direction='down',
                                                           gene_names=tdata.var_names,
                                                           n_jobs=-1)
                dt = pd.DataFrame({'Gene':dt.index,'EMD':dt['emd'],'Cell type':[i]*dt.shape[0]})
                # mann-whitney u, corrected p-values
                p = mwu(X,Y,tdata.var_names)
                dt = pd.merge(dt,p,how='left',on="Gene")
                dt = dt.loc[dt['EMD']<0,:] # take only 'down' 
                ngenes=500
                if np.sum(dt['pval_corrected']<0.01)<ngenes:
                    dt = dt.iloc[0:ngenes,:] # ranked from scprep
                    fc = log2aveFC(Y,X,dt['Gene'].to_list(),AnnData=tdata)
                    dt = pd.merge(dt,fc,how='left',on='Gene')
                else :
                    dt = dt.loc[dt['pval_corrected']<0.01,:] # may produce long rows
                    fc = log2aveFC(Y,X,dt['Gene'].to_list(),AnnData=tdata) # ave_diff= arg1 - arg2
                    dt = pd.merge(dt,fc,how='left',on='Gene')
                dge = dge.append(dt, ignore_index=True)
                print(dt.iloc[0:10,:])
                print('... computed in {:.2f}-s'.format(time.time()-start))
            dge.to_csv(os.path.join(pfp,'dge_'+t+'_'+fname+'_WTdown.csv'),index=False)
            print('---- WT down in {:.2f}-min'.format((time.time() - dge_total)/60))
        print('\n... {} done in {:.2f}-min'.format(t,(time.time()-t_total)/60))
        del tdata
    print('DGE finished in {:.2f}-min'.format((time.time()-dge_grandtotal)/60))


print('\n---\n')

print('Plotting embeddings...')

# plot 
if False :
    # aesthetics
    pal18=['#ee5264','#565656','#75a3b7','#ffe79e','#fac18a','#f1815f','#ac5861','#62354f','#2d284b','#f4b9b9','#c4bbaf',
           '#f9ebae','#aecef9','#aeb7f9','#f9aeae','#9c9583','#88bb92','#bde4a7','#d6e5e3']
    pal30 = sns.cubehelix_palette(30)
    # cmap_ctype={'Bergmann glia': '#f4b9b9',
    #  'Granule cells': '#bde4a7',
    #  'Astrocytes (II)': '#d6e5e3',
    #  'Astrocytes (I)': '#aeb7f9',
    #  'Unidentified': '#2d284b',
    #  'Unipolar brush cells': '#aecef9',
    #  'Inhibitory neuron (I)': '#9c9583',
    #  'Microglia': '#88bb92',
    #  'Excitatory neurons': '#ac5861',
    #  'Oligodendrocytes (I)': '#ffe79e',
    #  'Inhibitory neuron (III)': '#f9aeae',
    #  'Inhibitory neuron (II)': '#ee5264',
    #  'Oligodendrocytes (II)': '#f1815f',
    #  'Pericytes': '#75a3b7',
    #  'Vascular cells': '#ace5ee',
    #  'Purkinje cells': '#62354f',
    #  'OPC': '#fac18a',
    #  'Choroid plexus epithelial cells': '#c4bbaf'}
    cmap_genotype={'WT':'#010101',
                   'SCA1':'#ffd478'}
    pal5=['#f9aeae','#9c9583','#88bb92','#fac18a','#75a3b7']
    random.shuffle(pal5)
    sequential5 = sns.cubehelix_palette(5,start=.5,rot=-.75)
    if False :
        pal_timepoint=sns.color_palette('colorblind',len(adata.obs['timepoint'].unique()))
        cmap_timepoint={v:sequential5[i] for i,v in enumerate(adata.obs['timepoint'].unique())}
    # cmap_sample = {v:pal30[i] for i,v in enumerate(adata.obs['batch'].unique())} # if sample is names not nums
    cmap_sample = {v:pal30[i] for i,v in enumerate(np.unique(adata.obs.batch.astype(int).to_numpy()))}
    # color can get out of bounds so update range of colors so new one is called
    from matplotlib.colors import to_hex
    pal8=[to_hex(i,keep_alpha=True) for i in sns.cubehelix_palette(8)]
    pal26=['#ee5264','#565656','#75a3b7',
                                              '#ffe79e','#fac18a','#f1815f',
                                              '#ac5861','#62354f','#2d284b',
                                              '#f4b9b9','#c4bbaf','#f9ebae','#aecef9',
                                              '#aeb7f9','#f9aeae','#9c9583','#88bb92',
                                            '#bde4a7','#d6e5e3']+pal8
    pal46=pal18+sns.color_palette('dark',8)+sns.cubehelix_palette(12)+sns.color_palette('pastel',8)
    cmap_louvain = {v:pal46[i] for i,v in enumerate(np.unique(adata.obs.louvain.astype(int).to_numpy()))}
    cmap_ctype={'Granule cell': '#FAC18A',
      'Bergmann glia': '#AEB7F9',
      'Oligodendrocyte': '#75A3B7',
      'Astrocyte': '#F9AEAE',
      'GABAergic interneuron 1': '#F9EBAE',
      'GABAergic interneuron 2': '#88BB92',
      'Unipolar brush cell': '#BA61BA',
      'Endothelial cell': '#1C67EE',
      'Purkinje cell': '#EE5264',
      'Pericyte': '#2D284B',
      'Microglia': '#AC5861',
      'Oligodendrocyte progenitor cell': '#F1815F',
      'GABAergic interneuron 3': '#46A928'}


    fig,ax=plt.subplots(1,2,figsize=(4,3))
    scprep.plot.scatter2d(adata.obsm['X_umap'],
                          c=adata.obs['ctype'],
                          cmap=cmap_ctype,
                          ticks=None,
                          label_prefix='UMAP',
                          legend=False,
                          ax=ax[0],
                          s = 0.1,
                          alpha=0.6,
                          rasterized=True)
    scprep.plot.scatter2d(adata.obsm['X_phate'],
                          c=adata.obs['ctype'],
                          cmap=cmap_louvain,
                          ticks=None,
                          label_prefix='PHATE',
                          legend=True,
                          legend_loc=(1.01,0.2),
                          ax=ax[1],
                          s = 0.2,
                          alpha=0.6,
                          rasterized=True)
    fig.savefig(os.path.join(pfp,'embeds_mouse_ctype.pdf'),dpi=600, bbox_inches='tight')

print('Done with...')
print('    data processing,')
print('    cluster annotation figs,')
print('    DGE,')
print('    PHATE per genotype, and')
print('    data imputation')



'''
soultion to slow graph:

adj = adata.uns['neighbors']['connectivities'] + sparse.csr_matrix(np.diag(np.ones(adata.shape[0])))
phate_op = phate.PHATE(knn_dist='precomputed',
                           gamma=1,
                           n_jobs=-1,
                           random_state=rs)
x_phate=phate_op.fit_transform(adj)

col13=sns.color_palette('colorblind',13)
cmap_finalctype={v:col13[i] for i,v in enumerate(adata.obs['finalctype'].unique())}
scprep.plot.scatter2d(x_phate,
                      c=adata.obs['finalctype'],
                      cmap=cmap_finalctype,
                      legend_loc=(1.01,0.3),
                      title='w/',
                      ticks=None,
                      label_prefix='PHATE',
                      figsize=(8,4))

for MELD & MAGIC, can use
G = gt.Graph(adj,precomputed='adjacency')
meld_score = meld.MELD().fit_transform(G, adata.obs['RES'])
meld_score = meld_score - np.mean(meld_score) # mean center

G_withdata = gt.Graph(data=adj,use_pygsp=True,precomputed='adjacency')
G_withdata.knn_max = None
X_imputed = magic.MAGIC(knn_max=None).fit(X=adata.X,graph=G_withdata) # running fit_transform produces wrong shape
test = X_imputed.transform(adata.X,genes='all_genes')
'''
