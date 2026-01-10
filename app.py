import gradio as gr
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, list_repo_files 
import io
import os
import tempfile
import matplotlib
import numpy as np # NEW IMPORT for argmax
import xgboost as xgb # NEW IMPORT for DMatrix
import urllib
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder 

matplotlib.use("Agg")

# --- Configuration ---
REPO_ID = "mibrahimzia/exoplanet_model2.0" 
# ---------------------

# Load trained model from HF Hub
model_path = hf_hub_download(repo_id=REPO_ID, filename="exoplanet_model2.0.pkl")
# 'model' is an xgb.Booster object
model = joblib.load(model_path)

# Features (must match training order)
FEATURES = ["orbital_period", "transit_duration", "planet_radius", "star_temp", "star_radius"]

# MAPPING LABELS - MUST BE ZERO-INDEXED to match xgb.Booster output
# Based on LabelEncoder sorting of original labels: -1 -> 0, 0 -> 1, 1 -> 2
LABELS = {
    2: "✅ Confirmed Exoplanet", 
    1: "🟡 Candidate", 
    0: "❌ False Positive"
}
# Store the original label mapping for the distribution plot legend
ORIGINAL_LABELS = {-1: "❌ False Positive", 0: "🟡 Candidate", 1: "✅ Confirmed Exoplanet"}


# --- Utility Functions ---
def find_column(df, keywords):
    for col in df.columns:
        lname = col.lower()
        for kw in keywords:
            if kw in lname:
                return col
    return None

def normalize_and_prepare_df_for_metrics(df):
    """Detect columns, rename, drop NaNs, and create a binary 'y_true' column."""
    remap = {}
    c = find_column(df, ['koi_period','period','orbper','pl_orbper','tce_period'])
    if c: remap[c] = 'orbital_period'
    c = find_column(df, ['koi_duration','duration','transit_duration','tce_duration'])
    if c: remap[c] = 'transit_duration'
    c = find_column(df, ['koi_prad','prad','pl_rade','planet_radius','radius'])
    if c: remap[c] = 'planet_radius'
    c = find_column(df, ['koi_steff','st_teff','teff','star_temp'])
    if c: remap[c] = 'star_temp'
    c = find_column(df, ['koi_srad','st_rad','srad','star_radius'])
    if c: remap[c] = 'star_radius'
    c = find_column(df, ['koi_disposition','disposition','planet_disposition','status','kepflag','label'])
    if c: remap[c] = 'label'
    
    if remap:
        df = df.rename(columns=remap)
    
    # Drop rows where any required feature or the label is missing
    df = df.dropna(subset=FEATURES + ['label'], how='any')
    if df.empty:
        raise ValueError("No complete data points found after cleaning (Features and Label must be present).")
    
    # Target variable: Convert label to binary (1 for Confirmed, 0 otherwise)
    if 'label' in df.columns:
        def is_confirmed(x):
            if isinstance(x, (int, float)) and x == 1:
                return 1
            if isinstance(x, str) and 'CONFIRMED' in x.upper():
                return 1
            return 0
        df['y_true'] = df['label'].apply(is_confirmed)
    else:
        raise ValueError("The dataset does not contain a recognizable true 'label' column.")
    
    return df

# --- FIXED FUNCTION FOR MODEL METRICS (Part of Change #2) ---
# Added 'sample_name' as a new input
def plot_model_metrics(uploaded_file, sample_name):
    """
    Generates ROC curve and probability distribution plots for model evaluation.
    FIXED: Now handles both uploaded files and chosen sample files.
    """
    # 1. Load Data - START FIX
    path = None
    if uploaded_file is not None:
        path = getattr(uploaded_file, "name", None) or str(uploaded_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Uploaded file path not found: {path}")
    elif sample_name:
        repo_csv = sample_name
        try:
            path = hf_hub_download(repo_id=REPO_ID, filename=repo_csv)
        except Exception:
            raise FileNotFoundError(f"Sample file '{repo_csv}' not found. Upload a CSV or ensure it's in the repo.")
    else:
        # Default to kepler_exoplanet.csv from the repo for metrics
        repo_csv = "kepler_exoplanet.csv"
        try:
            path = hf_hub_download(repo_id=REPO_ID, filename=repo_csv)
        except Exception:
            raise FileNotFoundError(f"Default file '{repo_csv}' not found. Upload a CSV or ensure it's in the repo.")
    
    df = pd.read_csv(path)
    # 2. Prepare Data for Metrics (rest of function remains the same)
    
    # 2. Prepare Data for Metrics
    df = normalize_and_prepare_df_for_metrics(df)

    y_true = df['y_true']
    X = df[FEATURES]
    
    # CONVERSION FOR XGBOOST CORE API
    dmat = xgb.DMatrix(X) 
    
    # Predict probabilities (output is probas for the zero-indexed classes)
    probas = model.predict(dmat) 
    
    # The 'Confirmed' class is index 2 in our zero-indexed map: -1(0), 0(1), 1(2)
    confirmed_idx = 2
    
    if probas.shape[1] <= confirmed_idx:
        raise ValueError(f"Model prediction output has fewer than {confirmed_idx+1} classes. Check model training setup.")
        
    y_pred_proba_confirmed = probas[:, confirmed_idx]
    
    image_paths = []
    # 3. ROC Curve Plot (Accuracy Visualization)
    fig_roc, ax_roc = plt.subplots(figsize=(7, 7))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba_confirmed)
    roc_auc = auc(fpr, tpr)
    
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)')
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)')
    ax_roc.set_title('Model Performance: ROC Curve (AUC)')
    ax_roc.legend(loc="lower right")
    
    tmp_roc = os.path.join(tempfile.gettempdir(), "roc_curve.png")
    fig_roc.savefig(tmp_roc, bbox_inches="tight")
    plt.close(fig_roc)
    image_paths.append(tmp_roc)

    # 4. Probability Distribution Plot ("Range" Idea)
    fig_proba, ax_proba = plt.subplots(figsize=(7, 5))
    
    sns.histplot(y_pred_proba_confirmed[y_true == 1], bins=30, kde=True, color='green', 
                 label='Confirmed Exoplanets (True Label 1)', ax=ax_proba, stat="density", linewidth=0.5)
    sns.histplot(y_pred_proba_confirmed[y_true == 0], bins=30, kde=True, color='red', 
                 label='Not Confirmed (True Label 0/-1)', ax=ax_proba, stat="density", linewidth=0.5)
    
    ax_proba.axvline(x=0.5, color='blue', linestyle='--', label='Default Classification Threshold (0.5)')
    
    ax_proba.set_xlim(0, 1)
    ax_proba.set_xlabel('Predicted Probability of being a Confirmed Exoplanet')
    ax_proba.set_ylabel('Density')
    ax_proba.set_title('Model Confidence: Probability Distribution')
    ax_proba.legend()
    
    tmp_proba = os.path.join(tempfile.gettempdir(), "proba_dist.png")
    fig_proba.savefig(tmp_proba, bbox_inches="tight")
    plt.close(fig_proba)
    image_paths.append(tmp_proba)

    return image_paths

# --- Remaining Functions ---
def get_repo_csv_files():
    try:
        files = list_repo_files(repo_id=REPO_ID)
        csv_files = [f for f in files if f.endswith('.csv')]
        return csv_files
    except Exception as e:
        print(f"Error fetching repo files: {e}")
        return ["kepler_exoplanet.csv"]

def predict_single(orbital_period, transit_duration, planet_radius, star_temp, star_radius):
    """
    FIXED: Uses xgb.DMatrix and model.predict() with argmax for class label.
    """
    df = pd.DataFrame([[orbital_period, transit_duration, planet_radius, star_temp, star_radius]], columns=FEATURES)
    
    # 1. Create DMatrix
    dmat = xgb.DMatrix(df)
    
    # 2. Get probabilities (output is [p0, p1, p2])
    probas = model.predict(dmat) 
    
    # 3. Get the zero-indexed class with the highest probability
    pred_idx = np.argmax(probas, axis=1)[0]
    
    # 4. Map the zero-indexed result to the string label
    return LABELS.get(pred_idx, "Unknown")

def load_sample(sample_name):
    """
    FIXED: Ensures all error paths return the correct output types (DataFrame, File-path/None) 
    for the Gradio outputs, improving stability and user experience for sample data loading.
    """
    if not sample_name:
        # Return empty DataFrame and None for download file
        return pd.DataFrame(), None
        
    try:
        # Download the file
        sample_file_path = hf_hub_download(repo_id=REPO_ID, filename=sample_name)
        df = pd.read_csv(sample_file_path)
    except Exception as e:
        # If download/read fails, log the error and return empty/None to Gradio
        print(f"Error loading sample file '{sample_name}' from HF Hub: {e}")
        return pd.DataFrame(), None # FIXED: Must return None for the gr.File component in error case.

    if df.empty:
        print(f"Sample file '{sample_name}' is empty.")
        return pd.DataFrame(), None # Correct: (DataFrame, None)

    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        print(f"Sample file '{sample_name}' missing required features: {missing_cols}. Returning empty result.")
        return pd.DataFrame(), None # FIXED: Must return None for the gr.File component in error case.
        
    # --- Prediction Logic (No Change) ---
    dmat = xgb.DMatrix(df[FEATURES])
    probas = model.predict(dmat)
    preds_idx = np.argmax(probas, axis=1)
    df["prediction"] = [LABELS.get(p, "Unknown") for p in preds_idx]
    
    # Save the predicted DataFrame
    output_file = os.path.join(tempfile.gettempdir(), f"{os.path.splitext(sample_name)[0]}_predictions.csv")
    df.to_csv(output_file, index=False)

    return df, output_file # Correct: (DataFrame, File-path)

def predict_bulk(file):
    """
    FIXED: Uses xgb.DMatrix and model.predict() with argmax.
    """
    df = pd.read_csv(file.name)
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        return pd.DataFrame(), None
        
    # Prepare DMatrix
    dmat = xgb.DMatrix(df[FEATURES])
    
    # Get probabilities
    probas = model.predict(dmat)
    
    # Get the zero-indexed class with the highest probability
    preds_idx = np.argmax(probas, axis=1)

    # Map the zero-indexed result to the string label
    df["prediction"] = [LABELS.get(p, "Unknown") for p in preds_idx]
    
    output_file = os.path.join(tempfile.gettempdir(), "bulk_predictions.csv")
    df.to_csv(output_file, index=False)
    
    return df, output_file

# --- FIXED FUNCTION FOR FEATURE DISTRIBUTIONS (Part of Change #2) ---
# Added 'sample_name' as a new input
def plot_feature_distributions(uploaded_file, sample_name):
    def normalize_and_prepare_df_for_distributions(df):
        remap = {}
        c = find_column(df, ['koi_period','period','orbper','pl_orbper','tce_period'])
        if c: remap[c] = 'orbital_period'
        c = find_column(df, ['koi_duration','duration','transit_duration','tce_duration'])
        if c: remap[c] = 'transit_duration'
        c = find_column(df, ['koi_prad','prad','pl_rade','planet_radius','radius'])
        if c: remap[c] = 'planet_radius'
        c = find_column(df, ['koi_steff','st_teff','teff','star_temp'])
        if c: remap[c] = 'star_temp'
        c = find_column(df, ['koi_srad','st_rad','srad','star_radius'])
        if c: remap[c] = 'star_radius'
        c = find_column(df, ['koi_disposition','disposition','planet_disposition','status','kepflag','label'])
        if c: remap[c] = 'label'
        if remap:
            df = df.rename(columns=remap)
        needed = ['orbital_period','transit_duration','planet_radius','star_temp','star_radius','label']
        found = [c for c in needed if c in df.columns]
        if len(found) < 4:
            raise ValueError(f"Not enough recognizable columns for plotting. Found: {list(df.columns)}")
        if 'label' in df.columns:
            valid_labels = df['label'].dropna()
            if not valid_labels.empty:
                sample = valid_labels.iloc[0]
                # Use the ORIGINAL_LABELS map for the plot legend
                if isinstance(sample, (int, float)) and not isinstance(sample, str):
                    df['label_str'] = df['label'].map(ORIGINAL_LABELS).fillna("Unknown")
                else:
                    def norm_label(x):
                        xs = str(x).upper()
                        if 'CONFIRMED' in xs or '1' in xs: return "Confirmed Exoplanet"
                        if 'CANDID' in xs or '0' in xs: return "Candidate"
                        if 'FALSE' in xs or 'FP' in xs or '-1' in xs: return "False Positive"
                        return xs.title()
                    df['label_str'] = df['label'].apply(norm_label)
            else:
                df['label_str'] = "Unknown"
        else:
            df['label_str'] = "Unknown"
        df = df.dropna(subset=['orbital_period','transit_duration','planet_radius'], how='any')
        return df
        
    # 1. Load Data - START FIX
    path = None
    if uploaded_file is not None:
        path = getattr(uploaded_file, "name", None) or str(uploaded_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Uploaded file path not found: {path}")
    elif sample_name:
        repo_csv = sample_name
        try:
            path = hf_hub_download(repo_id=REPO_ID, filename=repo_csv)
        except Exception:
            raise FileNotFoundError(f"Sample file '{repo_csv}' not found. Upload a CSV or ensure it's in the repo.")
    else:
        repo_csv = "kepler_exoplanet.csv"
        try:
            path = hf_hub_download(repo_id=REPO_ID, filename=repo_csv)
        except Exception:
            raise FileNotFoundError(f"Default file '{repo_csv}' not found in repo. Upload a CSV first.")
    
    df = pd.read_csv(path)
    # 2. Prepare Data for Metrics (rest of function remains the same)
        
    df = normalize_and_prepare_df_for_distributions(df) # Call the local utility
    
    image_paths = []
    features_for_plot = ['orbital_period','transit_duration','planet_radius','star_temp','star_radius']
    for feat in features_for_plot:
        if feat not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6,4))
        try:
            q_low = df[feat].quantile(0.01)
            q_high = df[feat].quantile(0.99)
            df_filtered = df[(df[feat] > q_low) & (df[feat] < q_high)]
            sns.histplot(data=df_filtered, x=feat, hue='label_str', bins=40, kde=True, ax=ax, common_norm=False, multiple="stack")
            ax.set_xscale('log')
        except Exception as e:
            try:
                ax.hist([df[df['label_str']==g][feat].dropna() for g in df['label_str'].unique()], bins=40, label=df['label_str'].unique(), stacked=True)
                ax.legend()
            except Exception:
                df[feat].hist(bins=40, ax=ax)
                
        ax.set_title(f"{feat} distribution by class (98% data shown, log scale)")
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp.name, bbox_inches="tight")
        plt.close(fig)
        image_paths.append(tmp.name)
        
    if not image_paths:
        raise ValueError("No images were created. Check dataset columns.")
    return image_paths

def plot_feature_importance():
    """
    FIXED: Uses model.get_score() for feature importance (Core API equivalent).
    """
    # Use get_score for a Booster object
    importance = model.get_score(importance_type='gain')
    
    # The output is a dictionary mapping feature names to importance scores.
    # We must ensure all features are present, even if their score is 0.
    all_features_scores = {feat: importance.get(feat, 0) for feat in FEATURES}
    
    # Convert to lists for plotting
    feat_names = list(all_features_scores.keys())
    feat_importances = list(all_features_scores.values())
    
    if not feat_importances:
        raise AttributeError("Feature importances could not be calculated.")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=feat_importances, y=feat_names, ax=ax)
    ax.set_title("Feature Importance (Type: Gain)")
    ax.set_xlabel("Importance (Gain)")
    ax.set_ylabel("Feature")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

# --- Gradio UI Definition ---
REPO_CSV_FILES = get_repo_csv_files()

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## 🔭 Exoplanet Classifier – NASA Space Apps Challenge 2025")
        
        # --- START CHANGE #1: Persistent Note ---
        repo_link = f"https://huggingface.co/{REPO_ID}/tree/main"
        gr.Markdown("If you don't have formatted data **click [here](https://huggingface.co/spaces/mibrahimzia/2025_NASA_Space_Apps_ChallengeA_world_away_hunting_for_exoplanets_with_AI/tree/main)** to download few from space's repo or use the drop down menu to chose from.")
        # --- END CHANGE #1 ---
        
        with gr.Tab("Single Prediction"):
            gr.Interface(
                fn=predict_single,
                inputs=[
                    gr.Number(label="Orbital Period (days)"),
                    gr.Number(label="Transit Duration (hrs)"),
                    gr.Number(label="Planet Radius (Earth radii)"),
                    gr.Number(label="Star Temperature (K)"),
                    gr.Number(label="Star Radius (Solar radii)")
                ],
                outputs=gr.Label(label="Prediction")
            )
                        
        with gr.Tab("Bulk CSV Prediction"):
            # Defined here so they can be referenced by the 'Insights' tab buttons
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            preview = gr.Dataframe(label="Prediction Preview")
            download = gr.File(label="Download Predictions")
            btn = gr.Button("Run Bulk Classification (Upload)")
            btn.click(fn=predict_bulk, inputs=file_input, outputs=[preview, download])
                        
            gr.Markdown("---")
            gr.Markdown("Or **use a sample dataset** from the repository:")
            
            # Defined here so they can be referenced by the 'Insights' tab buttons
            sample_dropdown = gr.Dropdown(
                choices=REPO_CSV_FILES, 
                label="Choose a sample CSV dataset from the Hugging Face Space"
            )
            sample_preview = gr.Dataframe(label="Sample Prediction Preview")
            sample_download = gr.File(label="Download Sample Predictions")
            sample_btn = gr.Button("Classify Sample Data (Download & Run)")
            sample_btn.click(fn=load_sample, inputs=sample_dropdown, outputs=[sample_preview, sample_download])
                        
        with gr.Tab("Insights"):
            # NEW: Model Metrics Section
            gr.Markdown("### 📈 Model Accuracy & Exoplanet Probability Range")
            metrics_gallery = gr.Gallery(label="Model Metrics", show_label=False)
            metrics_btn = gr.Button("Generate Model Metrics (Requires True Labels)")
            # --- START CHANGE #2: Update inputs to include both file_input and sample_dropdown ---
            metrics_btn.click(fn=plot_model_metrics, inputs=[file_input, sample_dropdown], outputs=metrics_gallery)
            # --- END CHANGE #2 ---
            
            gr.Markdown("---")
            gr.Markdown("### 📊 Feature Distributions")
            viz_btn = gr.Button("Generate Feature Plots")
                            
            feature_gallery = gr.Gallery(label="Feature Ranges", show_label=False)
                            
            # --- START CHANGE #2: Update inputs to include both file_input and sample_dropdown ---
            viz_btn.click(fn=plot_feature_distributions, inputs=[file_input, sample_dropdown], outputs=feature_gallery)
            # --- END CHANGE #2 ---
                            
            gr.Markdown("### 🔎 Feature Importance")
            imp_btn = gr.Button("Show Importance")
            imp_plot = gr.Image(type="filepath")
            imp_btn.click(fn=plot_feature_importance, inputs=None, outputs=imp_plot)
            
    demo.launch()