import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import TruncatedSVD
import umap
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def run_integration(file_list, output_dir="static/plots", max_cells=10000, n_hvg=2000):
    """
    Memory-efficient batch integration for huge .h5ad files.

    Parameters:
    - file_list: list of .h5ad file paths
    - output_dir: directory to save UMAP plot
    - max_cells: maximum cells to sample per dataset
    - n_hvg: number of highly variable genes to select
    """
    if not file_list or len(file_list) < 2:
        raise ValueError("Need at least 2 datasets for batch integration.")

    adatas = []

    # Load datasets in backed mode and downsample
    for i, f in enumerate(file_list):
        ad = sc.read_h5ad(f, backed='r')
        n_sample = min(max_cells, ad.n_obs)
        np.random.seed(42)
        idx = np.random.choice(ad.n_obs, n_sample, replace=False)
        ad = ad[idx, :].to_memory()  # load only subset into memory
        ad.obs['batch'] = str(i)
        if ad.raw is None:
            ad.raw = ad.copy()
        adatas.append(ad)

    # Concatenate datasets
    adata = sc.concat(adatas, label='batch')

    # --- HVG selection ---
    # Use raw counts if available, else full AnnData
    if adata.raw is not None:
        hvg_input = sc.AnnData(
            X=adata.raw.X.copy(),
            obs=adata.obs.copy(),
            var=adata.raw.var.copy()
        )
    else:
        hvg_input = adata.to_memory()

    hvg_df = sc.pp.highly_variable_genes(
        hvg_input, flavor='seurat_v3', n_top_genes=n_hvg, batch_key='batch', inplace=False
    )
    adata.var['highly_variable'] = hvg_df['highly_variable']

    # Subset to HVGs
    adata = adata[:, adata.var.highly_variable]

    # Fill NaNs/Infs and convert sparse to ndarray
    X = adata.X
    if hasattr(X, "todense"):
        X = np.asarray(X.todense())  # convert to ndarray
    else:
        X = np.asarray(X)
    X = np.nan_to_num(X)

    # Remove zero-variance genes
    if X.shape[1] > 1:
        sel = VarianceThreshold(threshold=1e-8)
        X = sel.fit_transform(X)
        adata = adata[:, sel.get_support()]

    # Scale
    X = np.clip((X - X.mean(axis=0)) / X.std(axis=0), -10, 10)

    # --- TruncatedSVD ---
    n_components = min(30, X.shape[1]-1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)

    # --- UMAP ---
    umap_model = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(X_reduced)
    adata.obsm['X_umap'] = X_umap

    # Save UMAP plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "integration_umap.png")
    plt.figure(figsize=(6,6))
    batches = [int(b) for b in adata.obs['batch']]
    plt.scatter(X_umap[:,0], X_umap[:,1], c=batches, cmap='tab10', s=30)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Batch Integration UMAP")
    plt.colorbar(label="Batch")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    return plot_path
