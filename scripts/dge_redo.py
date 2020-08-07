

import os
import time
import datetime
import glob
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scprep
import graphtools as gt
import phate
from scipy import sparse
from scipy.stats import zscore 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.stats import mannwhitneyu, tiecorrect, rankdata
from statsmodels.stats.multitest import multipletests
import warnings
from adjustText import adjust_text
import sys
sys.path.append('/home/ngr4/project/scripts/')
import utils
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

pdfp = '/home/ngr4/project/scnd/data/processed'
pfp = '/home/ngr4/project/scnd/results/'

wtt = utils.load_adata(os.path.join(pdfp, 'mouse_wt_imputed.h5ad'))
mutt = utils.load_adata(os.path.join(pdfp, 'mouse_sca1_imputed.h5ad'))

fname = 'mouse_imp_by_genotype'

dge = pd.DataFrame()
for t in ['5wk', '12wk', '18wk', '24wk', '30wk',]:
    print('\nstarting timepoint {}\n'.format(t))
    start_t=time.time()
    
    group='ctype'
    for i in wtt.obs[group].unique() :
        start = time.time()
        
        print('  group: {}\n'.format(i))
                
        X = wtt[((wtt.obs[group]==i) & (wtt.obs['timepoint']==t)), :].layers['imputed']
        X_mut = mutt[((mutt.obs[group]==i) & (mutt.obs['timepoint']==t)), :].layers['imputed']
        
        X = np.asarray(X)
        X_mut = np.asarray(X_mut)
        
        print('    Ncells in X:{}'.format(X.shape[0]))
        print('    Ncells in X_mut:{}\n'.format(X_mut.shape[0]))

        p = utils.mwu(X,X_mut,wtt.var_names) # directionality doesn't matter
        emd = scprep.stats.differential_expression(X_mut,X,
                                                   measure = 'emd',
                                                   direction='both', 
                                                   gene_names=wtt.var_names,
                                                   n_jobs=-1)
        emd['Gene']=emd.index
        emd=emd.drop(columns='rank')
        fc = utils.log2aveFC(X_mut,X,wtt.var_names.to_list())
        gene_mismatch = fc['Gene'].isin(p['Gene'])
        if gene_mismatch.any():
            fc = fc.loc[gene_mismatch,:]
            warnings.warn('Warning: {} genes dropped due to p-val NA.'.format((gene_mismatch==False).sum()))
        dt = pd.merge(p,fc,how='left',on="Gene")
        gene_mismatch = emd['Gene'].isin(p['Gene'])
        if gene_mismatch.any():
            emd = emd.loc[gene_mismatch,:]
        dt = pd.merge(dt,emd,how='left',on='Gene')
        dt['Cell type']=[i]*dt.shape[0]
        dt['timepoint']=[str(t)]*dt.shape[0]
        dt['nlog10pvalcorrected']=(-1)*np.log10(dt['pval_corrected'])
        
        dge = dge.append(dt, ignore_index=True)
        print('... computed in {:.2f}-s'.format(time.time()-start))
    print('\nFinished timepoint {} in {:.2f}-min'.format(t,(time.time()-start_t)/60))
    
if True :
    # save volcano plot data
    dge.to_csv(os.path.join(pfp,'dge_'+fname+'_v2.csv'),index=False)
    if False:
        # bootstraps
        dge_min = dge.sort_values('pval').drop_duplicates(['Gene'])
        dge_min.to_csv(os.path.join(pfp,'dge_'+fname+'.csv'),index=False)

