import scanpy as sc
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def train_model(input_file, model_path="models/rf_model.pkl"):
    """
    Train RandomForest on adata.X with labels in adata.obs['cell_type'].
    Returns classification report as dataframe and saves model to model_path.
    """
    adata = sc.read(input_file)
    if "cell_type" not in adata.obs.columns:
        raise ValueError("Training dataset must have 'cell_type' in adata.obs")

    X = adata.X
    y = adata.obs["cell_type"].astype(str)

    # Flatten sparse matrices if needed
    try:
        X = X.toarray()
    except Exception:
        pass

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    return df_report
