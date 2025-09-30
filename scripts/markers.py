import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import pandas as pd

def compute_markers(input_file, output_dir="static/plots", n_genes=5):
    adata = sc.read(input_file)
    if "leiden" not in adata.obs.columns:
        from .clustering import run_clustering
        run_clustering(input_file, output_dir)
        adata = sc.read(input_file)

    sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
    result = sc.get.rank_genes_groups_df(adata, group=None)
    top = result.groupby('group').head(n_genes).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, "marker_genes.png")
    sc.pl.rank_genes_groups_heatmap(adata, n_genes=n_genes, groupby='leiden', show=False)
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close("all")

    return heatmap_path, top
