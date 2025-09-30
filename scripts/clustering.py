import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def run_clustering(input_file, output_dir="static/plots"):
    adata = sc.read(input_file)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    try:
        adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'mt-'))
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
        adata = adata[adata.obs.n_genes_by_counts < 2500, :]
        adata = adata[adata.obs.pct_counts_mt < 5, :]
    except Exception:
        pass

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    if adata.var.highly_variable.sum() > 0:
        adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "umap.png")
    sc.pl.umap(adata, color="leiden", show=False)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close("all")

    clusters = sorted(adata.obs["leiden"].unique().tolist())
    return plot_path, clusters, adata
