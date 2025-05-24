import os, pandas as pd, numpy as np, torch, collections, re, sys
root='pretrain_outputs/Lofgof_mESC_TF500+/pretrain'
expr_path='data/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500/BL--ExpressionData.csv'
# load
gate=pd.read_csv(os.path.join(root,'gate_values.csv'),index_col=0)
gate_mean=gate.mean(1)
print('Gate mean summary:\n', gate_mean.describe())

pred=pd.read_csv(os.path.join(root,'predictions_with_gene_names.csv'))
print('\nPrediction stats:')
print(pred[['Prediction','LogitVar']].describe())
print('Correlation pred vs var Pearson:', pred['Prediction'].corr(pred['LogitVar']))

# compute degree
genes=list(gate.index)
gene_to_idx={g:i for i,g in enumerate(genes)}
rows=[gene_to_idx.get(tf) for tf in pred['TF_gene'] if gene_to_idx.get(tf) is not None]
cols=[gene_to_idx.get(tg) for tg in pred['Target_gene'] if gene_to_idx.get(tg) is not None]
from collections import Counter
deg_counter=Counter(cols)

alpha=gate_mean
alpha_arr=np.array([alpha[g] for g in genes])
deg_arr=np.array([deg_counter.get(i,0) for i in range(len(genes))])
from scipy.stats import spearmanr
print('\nIn-degree summary:', pd.Series(deg_arr).describe())
print('Spearman(alpha,degree)=', spearmanr(alpha_arr, deg_arr)) 