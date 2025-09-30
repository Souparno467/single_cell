import scanpy as sc
import pandas as pd

def run_de_analysis(input_file, groupby="leiden", top_n=20):
    adata = sc.read(input_file)
    if groupby not in adata.obs.columns:
        from .clustering import run_clustering
        run_clustering(input_file)
        adata = sc.read(input_file)

    sc.tl.rank_genes_groups(adata, groupby, method="t-test")
    df = sc.get.rank_genes_groups_df(adata, group=None)
    top = df.groupby("group").head(top_n).reset_index(drop=True)
    return top
