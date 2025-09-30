import scanpy as sc
import joblib
import os

def predict_model(input_file, model_path="models/rf_model.pkl", output_file="uploads/pbmc3k_predicted.h5ad"):
    """
    Load model from model_path and predict labels for adata.X.
    Save annotated adata to output_file.
    Returns a dict of counts per predicted class.
    """
    adata = sc.read(input_file)
    clf = joblib.load(model_path)

    X = adata.X
    try:
        X = X.toarray()
    except Exception:
        pass

    preds = clf.predict(X)
    adata.obs["predicted_type"] = preds

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    adata.write(output_file)

    counts = adata.obs["predicted_type"].value_counts().to_dict()
    return counts
