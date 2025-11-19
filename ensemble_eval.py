import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, cohen_kappa_score,
    recall_score, precision_score, jaccard_score, hamming_loss,
    confusion_matrix, ConfusionMatrixDisplay
)
import models_vit as models
from util.datasets import build_dataset
from torch.utils.data import DataLoader, SequentialSampler

# ==========================================
# Configuration
# ==========================================
SEEDS = [42, 77, 123, 2025]
DATA_PATH = "/data/rishabhbhattad/data/CKD_Study/RETFound_data/CKDRET"
MODEL_ARCH = "RETFound_mae"
INPUT_SIZE = 224
NUM_CLASSES = 2
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_ROOT = "./output_dir"
CLASS_NAMES = ["Non-CKD", "CKD"]

def find_folder_for_seed(seed):
    prefix = f"retfound_mae_CKDRET_partial_s{seed}_"
    candidates = [d for d in os.listdir(OUTPUT_ROOT) if d.startswith(prefix)]
    if not candidates:
        raise ValueError(f"No output folder found for seed {seed}")
    candidates.sort()
    return os.path.join(OUTPUT_ROOT, candidates[-1])

def get_patient_id(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return base

def get_predictions(model, data_loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_filenames = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            
    if hasattr(data_loader.dataset, 'imgs'):
        all_filenames = [x[0] for x in data_loader.dataset.imgs]
    elif hasattr(data_loader.dataset, 'samples'):
        all_filenames = [x[0] for x in data_loader.dataset.samples]
    else:
        all_filenames = ["Unknown"] * len(data_loader.dataset)

    return np.concatenate(all_probs), np.concatenate(all_labels), all_filenames

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculates a dictionary of all relevant metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "F1 Score": f1_score(y_true, y_pred, average='macro'),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "Recall (Sensitivity)": recall_score(y_true, y_pred),
        "Specificity": specificity,
        "Precision": precision_score(y_true, y_pred),
        "Jaccard (IoU)": jaccard_score(y_true, y_pred),
        "Hamming Loss": hamming_loss(y_true, y_pred)
    }

def print_report(metrics, title):
    print(f"\n{'='*10} {title} {'='*10}")
    for name, value in metrics.items():
        print(f"{name:<20}: {value:.4f}")

def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved plot: {filename}")
    plt.close()

def main():
    print(f"Generating Full Report for seeds: {SEEDS}")
    
    # 1. Setup
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.data_path = DATA_PATH
    args.input_size = INPUT_SIZE
    args.dataratio = "1.0"
    args.stratified = False
    args.datasets_seed = 42
    
    dataset_test = build_dataset(is_train="test", args=args)
    sampler_test = SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=BATCH_SIZE, num_workers=4)
    
    # 2. Ensemble
    ensemble_probs = np.zeros((len(dataset_test), NUM_CLASSES))
    true_labels = None
    filenames = None

    for seed in SEEDS:
        folder = find_folder_for_seed(seed)
        ckpt_path = os.path.join(folder, "checkpoint-best.pth")
        if not os.path.exists(ckpt_path):
             # Double directory fix
             ckpt_path = os.path.join(folder, os.path.basename(folder), "checkpoint-best.pth")
        
        print(f"Processing Seed {seed}...")
        model = models.__dict__[MODEL_ARCH](num_classes=NUM_CLASSES, drop_path_rate=0.0, global_pool=True)
        model.to(DEVICE)
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
        model.load_state_dict(state_dict)
        
        probs, labels, fnames = get_predictions(model, data_loader_test, DEVICE)
        ensemble_probs += probs
        if true_labels is None:
            true_labels = labels
            filenames = fnames

    ensemble_probs /= len(SEEDS)
    
    # ==========================================
    # Level 1: Per-Image Analysis
    # ==========================================
    preds_img = np.argmax(ensemble_probs, axis=1)
    pos_probs_img = ensemble_probs[:, 1]
    
    metrics_img = calculate_metrics(true_labels, preds_img, pos_probs_img)
    print_report(metrics_img, "PER-IMAGE RESULTS")
    save_confusion_matrix(true_labels, preds_img, "Per-Image Confusion Matrix", "cm_ensemble_image.png")

    # ==========================================
    # Level 2: Per-Patient Analysis
    # ==========================================
    if filenames and filenames[0] != "Unknown":
        patient_preds = {}
        patient_labels = {}
        
        for prob, label, fname in zip(pos_probs_img, true_labels, filenames):
            pid = get_patient_id(fname)
            if pid not in patient_preds:
                patient_preds[pid] = []
                patient_labels[pid] = label
            patient_preds[pid].append(prob)
            
        final_probs = []
        final_labels = []
        for pid in patient_preds:
            final_probs.append(np.mean(patient_preds[pid]))
            final_labels.append(patient_labels[pid])
            
        final_probs = np.array(final_probs)
        final_labels = np.array(final_labels)
        final_preds = (final_probs > 0.5).astype(int)
        
        metrics_pat = calculate_metrics(final_labels, final_preds, final_probs)
        print_report(metrics_pat, "PER-PATIENT RESULTS")
        save_confusion_matrix(final_labels, final_preds, "Per-Patient Confusion Matrix", "cm_ensemble_patient.png")

if __name__ == "__main__":
    main()