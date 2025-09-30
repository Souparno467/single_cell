import os
from flask import Flask, render_template, send_file, Response, request
import pandas as pd
import numpy as np
import scanpy as sc
import joblib

from scripts.clustering import run_clustering
from scripts.markers import compute_markers
from scripts.train_model import train_model
from scripts.predict_model import predict_model
from scripts.integration import run_integration
from scripts.velocity import run_velocity
from scripts.de_analysis import run_de_analysis

# --- App setup ---
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["PLOT_FOLDER"] = "static/plots"
app.config["MODEL_FOLDER"] = "models"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PLOT_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)

# --- Auto-prepare demo datasets ---
PBMC_PATH = os.path.join(app.config["UPLOAD_FOLDER"], "pbmc3k.h5ad")
PBMC_LABELED_PATH = os.path.join(app.config["UPLOAD_FOLDER"], "pbmc3k_labeled.h5ad")

if not os.path.exists(PBMC_PATH):
    adata_demo = sc.datasets.pbmc3k()
    adata_demo.write(PBMC_PATH)

if not os.path.exists(PBMC_LABELED_PATH):
    ad = sc.read(PBMC_PATH)
    ad.obs["cell_type"] = np.random.choice(["T cell", "B cell", "Monocyte"], size=ad.n_obs)
    ad.write(PBMC_LABELED_PATH)

MODEL_FILE = os.path.join(app.config["MODEL_FOLDER"], "rf_model.pkl")

# --- Routes ---
@app.route("/")
def index():
    model_trained = os.path.exists(MODEL_FILE)
    return render_template("index.html", model_trained=model_trained)

@app.route("/unsupervised")
def unsupervised():
    plot_path, clusters, adata = run_clustering(PBMC_PATH, app.config["PLOT_FOLDER"])
    clustered_path = os.path.join(app.config["UPLOAD_FOLDER"], "pbmc3k_clustered.h5ad")
    adata.write(clustered_path)
    return render_template("results_unsupervised.html",
                           plot_path=os.path.basename(plot_path),
                           clusters=clusters)

@app.route("/markers")
def markers():
    clustered_path = os.path.join(app.config["UPLOAD_FOLDER"], "pbmc3k_clustered.h5ad")
    if not os.path.exists(clustered_path):
        run_clustering(PBMC_PATH, app.config["PLOT_FOLDER"])
    heatmap_path, markers_df = compute_markers(clustered_path, app.config["PLOT_FOLDER"])
    return render_template("results_markers.html",
                           heatmap_path=os.path.basename(heatmap_path),
                           markers_html=markers_df.to_html(classes="table table-striped"))

@app.route("/train")
def train():
    try:
        report_df = train_model(PBMC_LABELED_PATH, MODEL_FILE)
        return render_template("results_supervised.html",
                               report=report_df.to_html(classes="table table-striped"),
                               model_trained=True)
    except Exception as e:
        return render_template("error.html", message=str(e))

@app.route("/predict")
def predict():
    if not os.path.exists(MODEL_FILE):
        return render_template("error.html", message="Model not trained yet. Run /train first.")
    results = predict_model(PBMC_PATH, MODEL_FILE,
                            output_file=os.path.join(app.config["UPLOAD_FOLDER"], "pbmc3k_predicted.h5ad"))
    df = pd.DataFrame.from_dict(results, orient="index", columns=["count"])
    return render_template("results_supervised.html",
                           report=df.to_html(classes="table table-striped"),
                           model_trained=True)

# --- Batch Integration ---
@app.route("/integration", methods=["GET", "POST"])
def integration_route():
    if request.method == "POST":
        uploaded_files = request.files.getlist("files")
        file_paths = []
        for f in uploaded_files:
            if f.filename.endswith(".h5ad"):
                path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
                f.save(path)
                file_paths.append(path)

        if len(file_paths) < 2:
            return render_template("error.html", message="Upload at least 2 .h5ad files.")

        try:
            plot_path = run_integration(file_paths, app.config["PLOT_FOLDER"])
            return render_template("results_unsupervised.html",
                                   plot_path=os.path.basename(plot_path),
                                   clusters=["integration_demo"])
        except Exception as e:
            return render_template("error.html", message=str(e))

    return render_template("upload_integration.html")

# --- Velocity ---
@app.route("/velocity")
def velocity_route():
    ok, out = run_velocity(PBMC_PATH, app.config["PLOT_FOLDER"])
    if not ok:
        return render_template("error.html", message=out)
    return render_template("results_unsupervised.html",
                           plot_path=os.path.basename(out),
                           clusters=["velocity_demo"])

# --- Differential expression ---
@app.route("/de")
def de_route():
    clustered_path = os.path.join(app.config["UPLOAD_FOLDER"], "pbmc3k_clustered.h5ad")
    if not os.path.exists(clustered_path):
        run_clustering(PBMC_PATH, app.config["PLOT_FOLDER"])
    de_df = run_de_analysis(clustered_path)
    return render_template("results_supervised.html", report=de_df.to_html(classes="table table-striped"))

# --- CSV Downloads ---
@app.route("/download_clusters")
def download_clusters():
    clustered_path = os.path.join(app.config["UPLOAD_FOLDER"], "pbmc3k_clustered.h5ad")
    if not os.path.exists(clustered_path):
        return render_template("error.html", message="Run /unsupervised first.")
    adata = sc.read(clustered_path)
    df = adata.obs[["leiden"]].copy()
    csv_data = df.to_csv(index=True)
    return Response(csv_data, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=clusters.csv"})

@app.route("/download_predictions")
def download_predictions():
    pred_path = os.path.join(app.config["UPLOAD_FOLDER"], "pbmc3k_predicted.h5ad")
    if not os.path.exists(pred_path):
        return render_template("error.html", message="Run /predict first.")
    adata = sc.read(pred_path)
    df = adata.obs[["predicted_type"]].copy()
    csv_data = df.to_csv(index=True)
    return Response(csv_data, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=predictions.csv"})

@app.route("/plots/<filename>")
def plots_static(filename):
    return send_file(os.path.join(app.config["PLOT_FOLDER"], filename))

if __name__ == "__main__":
    app.run(debug=True)
