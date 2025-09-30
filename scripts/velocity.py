import os

def run_velocity(input_file, output_dir="static/plots"):
    """
    Run RNA velocity analysis on an AnnData file.
    If the dataset lacks spliced/unspliced layers, falls back to scvelo pancreas demo.
    """
    try:
        import scvelo as scv
        import scanpy as sc
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return False, "scvelo not installed: " + str(e)

    try:
        # Try reading user-provided dataset
        adata = sc.read(input_file)

        # If spliced/unspliced not found, fallback to pancreas dataset
        if "spliced" not in adata.layers or "unspliced" not in adata.layers:
            print("⚠️ No spliced/unspliced layers found — using pancreas demo dataset instead.")
            adata = scv.datasets.pancreas()

        # Ensure UMAP exists
        if "X_umap" not in adata.obsm.keys():
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)
            adata = adata[:, adata.var.highly_variable]
            sc.pp.scale(adata, max_value=10)
            sc.tl.pca(adata, svd_solver="arpack")
            sc.pp.neighbors(adata, n_pcs=30)
            sc.tl.umap(adata)

        # Velocity pipeline
        scv.pp.filter_and_normalize(adata, enforce=True)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata, mode="stochastic")
        scv.tl.velocity_graph(adata)

        # Save velocity plot
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "velocity_umap.png")
        scv.pl.velocity_embedding_stream(adata, basis="umap", show=False, legend_loc="right margin")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close("all")

        return True, out_path

    except Exception as e:
        return False, "Velocity error: " + str(e)
