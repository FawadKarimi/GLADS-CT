
import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/LungDxFormer/src")
sys.path.append("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched")

from dataset import LIDCSliceDataset
from lungdxformer.models.lungdxformer import LungDxFormer
from glads_slice_model import GLADS_CT_Slice

# Ensure GPU 1 is selected
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RUNS_DIR = "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/runs"
os.makedirs(RUNS_DIR, exist_ok=True)
CKPT_DIR = "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

df = pd.read_csv("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/splits_paper_matched.csv")

# Class weights from TRAIN set
train_df = df[df["split_slice"] == "train"]
val_df = df[df["split_slice"] == "val"]

counts = train_df["label"].value_counts().sort_index().values
total_samples = len(train_df)
class_weights = torch.tensor([total_samples / (3.0 * c) for c in counts], dtype=torch.float32, device=device)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

runs_to_execute = [
    # Baseline LungDxFormer B0 (Paper Match)
    {"id": "LDF-B0-seed42", "model": "LungDxFormer", "seed": 42, "obj": "CE"},
    {"id": "LDF-B0-seed43", "model": "LungDxFormer", "seed": 43, "obj": "CE"},
    {"id": "LDF-B0-seed44", "model": "LungDxFormer", "seed": 44, "obj": "CE"},
    # GLADS-CT Gated Hard (Ablation: Gated fusion without soft targets)
    {"id": "GLADS-Hard-seed42", "model": "GLADS", "seed": 42, "obj": "CE"},
    {"id": "GLADS-Hard-seed43", "model": "GLADS", "seed": 43, "obj": "CE"},
    {"id": "GLADS-Hard-seed44", "model": "GLADS", "seed": 44, "obj": "CE"},
    # GLADS-CT Full (Gated fusion + Disagreement-Aware Supervision)
    {"id": "GLADS-Disag-seed42", "model": "GLADS", "seed": 42, "obj": "DISAG"},
    {"id": "GLADS-Disag-seed43", "model": "GLADS", "seed": 43, "obj": "DISAG"},
    {"id": "GLADS-Disag-seed44", "model": "GLADS", "seed": 44, "obj": "DISAG"},
]

all_run_summaries = {}

for run in runs_to_execute:
    run_id = run["id"]
    model_type = run["model"]
    seed = run["seed"]
    obj = run["obj"]
    
    print(f"\n==========================================")
    print(f"Starting Run: {run_id} (Model: {model_type}, Seed: {seed}, Obj: {obj})")
    print(f"==========================================")
    set_seed(seed)
    
    train_loader = DataLoader(LIDCSliceDataset(train_df, is_train=True), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(LIDCSliceDataset(val_df, is_train=False), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    if model_type == "LungDxFormer":
        model = LungDxFormer(image_size=64, num_classes=3, fusion_type="concat").to(device)
    else:
        model = GLADS_CT_Slice(image_size=64, num_classes=3).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=False)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []
    
    start_time = time.time()
    
    for epoch in range(1, 101):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            rt = batch["reader_target"].to(device, non_blocking=True)
            disag = batch["disagreement"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(x)
                logits = out["logits"]
                
                if obj == "DISAG":
                    u = disag.unsqueeze(1)
                    target = (1.0 - u) * F.one_hot(y, 3).float() + u * rt
                    mass = target * class_weights.unsqueeze(0)
                    log_d = F.log_softmax(logits, dim=1)
                    loss = -(mass * log_d).sum() / mass.sum().clamp_min(1e-8)
                else:
                    loss = F.cross_entropy(logits, y, weight=class_weights)
                    
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * len(y)
            
        train_loss /= len(train_df)
        
        # Validation on frozen VAL partition
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    out = model(x)
                    logits = out["logits"]
                    loss = F.cross_entropy(logits, y, weight=class_weights)
                val_loss += loss.item() * len(y)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y.cpu().numpy())
                
        val_loss /= len(val_df)
        val_acc = accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        f1_per_class = f1_score(val_targets, val_preds, average=None, zero_division=0)
        
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            ckpt_path = os.path.join(CKPT_DIR, f"{run_id}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_f1_classes": f1_per_class.tolist()
            }, ckpt_path)
        else:
            patience_counter += 1
            
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "f1_benign": float(f1_per_class[0]),
            "f1_indeterminate": float(f1_per_class[1]),
            "f1_malignant": float(f1_per_class[2]),
            "lr": lr,
            "is_best": is_best
        }
        history.append(epoch_record)
        
        if epoch % 5 == 0 or is_best:
            print(f"Ep {epoch:03d} | TrLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | ValAcc: {val_acc*100:.2f}% | ValMacroF1: {val_macro_f1*100:.2f}% | F1(B/I/M): {f1_per_class[0]*100:.1f}/{f1_per_class[1]*100:.1f}/{f1_per_class[2]*100:.1f} {'[*BEST*]' if is_best else ''}")
            
        if patience_counter >= 10:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with Val Loss {best_val_loss:.4f}.")
            break
            
    total_time = time.time() - start_time
    
    # Load best checkpoint info
    best_ckpt = torch.load(os.path.join(CKPT_DIR, f"{run_id}_best.pt"))
    summary = {
        "run_id": run_id,
        "model": model_type,
        "seed": seed,
        "objective": obj,
        "best_epoch": best_ckpt["epoch"],
        "total_epochs": epoch,
        "total_seconds": total_time,
        "val_loss": best_ckpt["val_loss"],
        "val_accuracy": best_ckpt["val_acc"],
        "val_macro_f1": best_ckpt["val_macro_f1"],
        "val_f1_benign": best_ckpt["val_f1_classes"][0],
        "val_f1_indeterminate": best_ckpt["val_f1_classes"][1],
        "val_f1_malignant": best_ckpt["val_f1_classes"][2],
    }
    all_run_summaries[run_id] = summary
    
    # Save history
    with open(os.path.join(RUNS_DIR, f"{run_id}_history.json"), "w") as f:
        json.dump({"summary": summary, "history": history}, f, indent=2)

print("\n==========================================")
print("ALL MULTI-SEED RUNS COMPLETE!")
print("==========================================")
print(json.dumps(all_run_summaries, indent=2))
with open("/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/multi_seed_summaries.json", "w") as f:
    json.dump(all_run_summaries, f, indent=2)
