import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests
from scanpy import read
from random import sample,choices

from spatialDEG.find_cluster_degs import two_group_degs

#test
A = np.array([
[0, 0, 13, 0, 15],
[0, 22, 23, 24, 25],
[31, 32, 33, 34, 35],
[0, 0, 43, 44, 45],
[1, 52, 53, 54, 55]])
np.shape(A)
s,p = mannwhitneyu(A[:2,0],A[3:4,0])
column_names = ['gene1','gene2','gene3','gene4','gene5']
row_names = ['c1','c2','c3','c4','c5']
df = pd.DataFrame(A,columns=column_names,index=row_names)
#
all_vals = A[:,0]
test_vals = all_vals[0:3]
control_vals = all_vals[3:]
mannwhitneyu(test_vals,control_vals)
#
test_mean = np.mean(test_vals, axis=0) + 1e-9
control_mean = np.mean(control_vals, axis=0) + 1e-9
log2fc = np.log2(test_mean/control_mean + 10e-5)

## adata
adata = read('/home/zuolulu/spatialDEG/results.h5ad')
adata.obs['cluster'] = choices(range(10),3639)
adata.write('/home/zuolulu/spatialDEG/results.h5ad')


if type(control_groups) == str:
        control_groups = [control_groups]


control_groups = list(range(1,10+1))
test_cells, control_cells = (
        adata.obs['group'] == 0,
        adata.obs['group'].isin(list(range(1,6+1))),
    )

num_test_cells = test_cells.sum()
num_groups = len(control_groups)

X_data = adata.X
genes = adata.var_names


import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import issparse
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.spatial import distance
from statsmodels.sandbox.stats.multicomp import multipletests
from scanpy import read
from collections import Counter
#from concurrent.futures import ThreadPoolExecutor as PoolExecutor


def find_all_markers(
    adata,
    group,
    genes=None,
    layer=None,
    X_data=None,
):
    """Find marker genes for each group of spots based on gene expression.    
    
    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        genes: `list` or None (default: `None`)
            The list of genes that will be used to subset the data for dimension
            reduction and clustering. If `None`, all genes will be used.
        layer: `str` or None (default: `None`)
            The layer that will be used to retrieve data for dimension reduction
            and clustering. If `None`, .X is used.
        group: `str` or None (default: `None`)
            The column key/name that identifies the grouping information (for 
            example, clusters that correspond to different cell types) of spots. 
            This will be used for calculating group-specific genes.
        test_group: `str` or None (default: `None`)
            The group name from `group` for which markers has to be found.
        control_groups: `list`
            The list of group name(s) from `group` for which markers has to be 
            tested against.
        X_data: `np.ndarray` (default: `None`)
            The user supplied data that will be used for marker gene detection 
            directly.
    -------
        Returns an updated `~anndata.AnnData` with a new property `cluster_markers`
        in the .uns attribute, which includes a concated pandas DataFrame of the 
        differential expression analysis result for all groups and a dictionary 
        where keys are cluster numbers and values are lists of marker genes for 
        the corresponding clusters.
    """
    X_data = adata.X
    if genes is not None:
        genes = genes
    else:
        genes = adata.var_names
    
    if group not in adata.obs.keys():
        raise ValueError(f"group {group} is not a valid key for .obs in your adata object.")
    else:
        adata.obs[group] = adata.obs[group].astype("str")
        cluster_set = np.sort(adata.obs[group].unique())
        
    if len(cluster_set) < 2:
        raise ValueError(f"the number of groups for the argument {group} must be at least two.")
    
    de_tables = [None] * len(cluster_set)
    de_genes = {}

    if len(cluster_set) > 2:
        for i, test_group in enumerate(cluster_set):
            control_groups = sorted(set(cluster_set).difference([test_group]))
             
            de = find_markers(
                adata,
                test_group,  # difference in find_all_markers : control test_group
                control_groups,
                genes=genes,
                X_data=X_data,
                group=group,
            )

            de_tables[i] = de.copy()
            de_genes[i] = [k for k, v in Counter(de["gene"]).items() if v >= 1]
    else:
        de = find_markers(
            adata,
            cluster_set[0],
            cluster_set[1],
            genes=genes,
            X_data=X_data,
            group=group,
        )

        de_tables[0] = de.copy()
        de_genes[0] = [k for k, v in Counter(de["gene"]).items() if v >= 1]

    de_table = pd.concat(de_tables).reset_index().drop(columns=["index"])
    
    adata.uns["cluster_markers"] = {"deg_table": de_table, "de_genes": de_genes}

    return adata



def find_markers_mod(
    adata,
    test_group,  ## difference in find_all_markers : control test_group
    control_groups,
    genes=None,
    layer=None,
    X_data=None,
    group=None,
    qval_thresh=0.05,
    ratio_expr_thresh=0.1,
    diff_ratio_expr_thresh=0,
    log2fc_thresh=0,
):
    """Find marker genes between two groups of spots based on gene expression.
    Test each gene for differential expression between spots in one group 
    and another groups via Mann-Whitney U test. we calcute the percentage of 
    spots expressing the gene in the test group(ratio_expr), the difference 
    between the percentages of spots expressing the gene in the test group and 
    control group,the expression fold change between the test and control group
    (log2fc),in addition, qval is calculated using Benjamini-Hochberg.
    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        genes: `list` or None (default: `None`)
            The list of genes that will be used to subset the data for dimension
            reduction and clustering. If `None`, all genes will be used.
        layer: `str` or None (default: `None`)
            The layer that will be used to retrieve data for dimension reduction
            and clustering. If `None`, .X is used.
        group: `str` or None (default: `None`)
            The column key/name that identifies the grouping information (for 
            example, clusters that correspond to different cell types) of spots. 
            This will be used for calculating group-specific genes.
        test_group: `str` or None (default: `None`)
            The group name from `group` for which markers has to be found.
        control_groups: `list`
            The list of group name(s) from `group` for which markers has to be 
            tested against.
        X_data: `np.ndarray` (default: `None`)
            The user supplied data that will be used for marker gene detection 
            directly.
        qval_thresh: `float` (default: 0.05)
            The maximal threshold of qval to be considered as significant genes.
        ratio_expr_thresh: `float` (default: 0.1)
            The minimum percentage of spots expressing the gene in the test group.
        diff_ratio_expr_thresh: `float` (default: 0)
            The minimum of the difference between two groups.
        log2fc: `float` (default: 0)
            The minimum expression log2 fold change.
    Returns
    -------
        A pandas DataFrame of the differential expression analysis result between
        the two groups.
    """
    if X_data is not None:
        X_data = X_data
    else:
        X_data = adata.X
    
    if genes is not None:
        genes = genes
    else:
        genes = adata.var_names   

    #n_spots, n_genes = X_data.shape
    sparse = issparse(X_data)

    if type(control_groups) == str:
        control_groups = [control_groups]
    num_groups = len(control_groups)
    
    test_cells = adata.obs[group] == test_group
    control_cells = adata.obs[group].isin(control_groups)
    num_test_cells = test_cells.sum()
    num_control_cells = control_cells.sum()

    de = []
    for i_gene, gene in tqdm(enumerate(genes), desc="identifying top markers for each group"): 

        all_vals = X_data[:, i_gene].A if sparse else X_data[:, i_gene]
        test_vals = all_vals[test_cells]
        control_vals =all_vals[control_cells]
        
        ## ratio_expr
        ratio_expr = len(test_vals.nonzero()[0])/num_test_cells
        if ratio_expr < ratio_expr_thresh:
            continue
        
        ## log2fc
        test_mean = test_vals.mean() + 1e-9
        control_mean = control_vals.mean() + 1e-9
        log2fc = np.log2(test_mean/control_mean + 10e-5)
            
        ## pvals
        if len(control_vals.nonzero()[0]) > 0:
            pvals = mannwhitneyu(test_vals, control_vals)[1][0]
        else: 
            pvals = 1
            
        ## diff_ratio_expr
        diff_ratio_expr = ratio_expr - len(control_vals.nonzero()[0])/num_control_cells

        de.append(
            (   
                gene,
                control_groups[i],
                log2fc,
                pvals,
                ratio_expr,
                diff_ratio_expr,
            )
        )

    de = pd.DataFrame(
        de,
        columns=[
            "gene",
            "versus_group",
            "log2fc",
            "pval",
            "ratio_expr",
            "diff_ratio_expr",
        ],
    )
    
    if de.shape[0] > 1:
        de["qval"] = multipletests(de["pval"].values, method="fdr_bh")[1]
    else:
        de["qval"] = [np.nan for _ in range(de.shape[0])]
    de["test_group"] = [test_group for _ in range(de.shape[0])]
    out_order = [
        "gene",
        "test_group",
        "versus_group",
        "ratio_expr",
        "diff_ratio_expr",
        "log2fc",
        "pval",
        "qval",
    ]
    de = de[out_order].sort_values(by="qval")
    de = de[(de.qval < qval_thresh)&(de.diff_ratio_expr > diff_ratio_expr_thresh)&(de.log2fc> log2fc_thresh)].reset_index(drop=True)
    
    return de

def find_markers(
    adata,
    test_group,  ## difference in find_all_markers : control test_group
    control_groups,
    genes=None,
    layer=None,
    X_data=None,
    group=None,
    qval_thresh=0.05,
    ratio_expr_thresh=0.1,
    diff_ratio_expr_thresh=0,
    log2fc_thresh=0,
):
    """Find marker genes between two groups of spots based on gene expression.
    Test each gene for differential expression between spots in one group 
    and another groups via Mann-Whitney U test. we calcute the percentage of 
    spots expressing the gene in the test group(ratio_expr), the difference 
    between the percentages of spots expressing the gene in the test group and 
    control group,the expression fold change between the test and control group
    (log2fc),in addition, qval is calculated using Benjamini-Hochberg.
    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        genes: `list` or None (default: `None`)
            The list of genes that will be used to subset the data for dimension
            reduction and clustering. If `None`, all genes will be used.
        layer: `str` or None (default: `None`)
            The layer that will be used to retrieve data for dimension reduction
            and clustering. If `None`, .X is used.
        group: `str` or None (default: `None`)
            The column key/name that identifies the grouping information (for 
            example, clusters that correspond to different cell types) of spots. 
            This will be used for calculating group-specific genes.
        test_group: `str` or None (default: `None`)
            The group name from `group` for which markers has to be found.
        control_groups: `list`
            The list of group name(s) from `group` for which markers has to be 
            tested against.
        X_data: `np.ndarray` (default: `None`)
            The user supplied data that will be used for marker gene detection 
            directly.
        qval_thresh: `float` (default: 0.05)
            The maximal threshold of qval to be considered as significant genes.
        ratio_expr_thresh: `float` (default: 0.1)
            The minimum percentage of spots expressing the gene in the test group.
        diff_ratio_expr_thresh: `float` (default: 0)
            The minimum of the difference between two groups.
        log2fc: `float` (default: 0)
            The minimum expression log2 fold change.
    Returns
    -------
        A pandas DataFrame of the differential expression analysis result between
        the two groups.
    """
    if X_data is not None:
        X_data = X_data
    else:
        X_data = adata.X
    
    if genes is not None:
        genes = genes
    else:
        genes = adata.var_names   

    #n_spots, n_genes = X_data.shape
    sparse = issparse(X_data)

    if type(control_groups) == str:
        control_groups = [control_groups]
    num_groups = len(control_groups)
    test_cells = adata.obs[group] == test_group
    num_test_cells = test_cells.sum()

    de = []
    for i_gene, gene in tqdm(enumerate(genes), desc="identifying top markers for each group"): 

        all_vals = X_data[:, i_gene].A if sparse else X_data[:, i_gene]
        test_vals = all_vals[test_cells]
        test_mean = test_vals.mean() + 1e-9
        ## ratio_expr
        ratio_expr = len(test_vals.nonzero()[0])/num_test_cells
        if ratio_expr < ratio_expr_thresh:
            continue
        
        for i in range(num_groups):
            control_vals = all_vals[adata.obs[group] == control_groups[i]]
            
            ## log2fc
            control_mean = np.mean(control_vals, axis=0) + 1e-9
            log2fc = np.log2(test_mean/control_mean + 10e-5)[0]
            
            ## pvals
            if len(control_vals.nonzero()[0]) > 0:
                pvals = mannwhitneyu(test_vals, control_vals)[1][0]
            else: 
                pvals = 1
            
            ## diff_ratio_expr
            diff_ratio_expr = ratio_expr - len(control_vals.nonzero()[0])/len(control_vals)

            de.append(
                (   
                    gene,
                    control_groups[i],
                    log2fc,
                    pvals,
                    ratio_expr,
                    diff_ratio_expr,
                )
            )

    de = pd.DataFrame(
        de,
        columns=[
            "gene",
            "versus_group",
            "log2fc",
            "pval",
            "ratio_expr",
            "diff_ratio_expr",
        ],
    )
    
    if de.shape[0] > 1:
        de["qval"] = multipletests(de["pval"].values, method="fdr_bh")[1]
    else:
        de["qval"] = [np.nan for _ in range(de.shape[0])]
    de["test_group"] = [test_group for _ in range(de.shape[0])]
    out_order = [
        "gene",
        "test_group",
        "versus_group",
        "ratio_expr",
        "diff_ratio_expr",
        "log2fc",
        "pval",
        "qval",
    ]
    de = de[out_order].sort_values(by="qval")
    de = de[(de.qval < qval_thresh)&(de.diff_ratio_expr > diff_ratio_expr_thresh)&(de.log2fc> log2fc_thresh)].reset_index(drop=True)
    
    return de

def find_spatial_cluster_markers(
    adata, 
    test_group, 
    control_group, 
    x_name, 
    y_name,
    ratio_expr_thresh=0.1,
    diff_ratio_expr_thresh=0,
    log2fc_thresh=0,
    ):
    x=adata.obs[x_name].tolist()
    y=adata.obs[y_name].tolist()
    X=np.array([x, y]).T.astype(np.float32)
    adj = distance.cdist(X, X, 'euclidean')
	start, end= np.quantile(adj[adj!=0],q=0.001), np.quantile(adj[adj!=0],q=0.1)
	r=search_radius(test_group=target, cell_id=adata.obs.index.tolist(), x=adata.obs[x_name].tolist(), y=adata.obs[y_name].tolist(), group=adata.obs[domain_name].tolist(), start=start, end=end, num_min=10, num_max=14,  max_run=100)
	nbr_domians=find_neighbor_clusters(test_group=target,
								   cell_id=adata.obs.index.tolist(), 
								   x=adata.obs[x_name].tolist(), 
								   y=adata.obs[y_name].tolist(), 
								   group=adata.obs[domain_name].tolist(), 
								   radius=r,
								   ratio=1/2)
	nbr_domians=nbr_domians[0:3]
	de_genes_info=rank_genes_groups(input_adata=adata,
								test_group=target,
								nbr_list=nbr_domians, 
								label_col=domain_name, 
								adj_nbr=True, 
								log=True)
	de_genes_info=de_genes_info[(de_genes_info["pvals_adj"]<0.05)]
	filtered_info=de_genes_info
	filtered_info=filtered_info[(filtered_info["pvals_adj"]<0.05) &
							(filtered_info["in_out_group_ratio"]>min_in_out_group_ratio) &
							(filtered_info["in_group_fraction"]>min_in_group_fraction) &
							(filtered_info["fold_change"]>min_fold_change)]
	filtered_info=filtered_info.sort_values(by="in_group_fraction", ascending=False)
	filtered_info["target_dmain"]=target
	filtered_info["neighbors"]=str(nbr_domians)
	print("SVGs for domain ", str(target),":", filtered_info["genes"].tolist())
	return filtered_info