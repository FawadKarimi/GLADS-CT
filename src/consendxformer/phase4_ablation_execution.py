"""Single ablation execution, reusing unchanged Phase 4 training primitives."""
from __future__ import annotations

import math

import torch

from .phase4_execution import build_phase4_initial_datasets, _model, _loader, _run_epoch, _build_test_dataset
from .phase4_training import compute_training_class_weights
from .phase4_ablation_contracts import (
    ID_FIELDS, CLASS_ORDER, identity_rows, validate_identity, checkpoint_payload, save_checkpoint,
    load_checkpoint_pair, build_predictions,
)
from .reproducibility import set_global_seed


class ExactEvalLoader:
    """Preserve the underlying loader while checking every yielded identity/label/path."""
    def __init__(self, loader, frame):
        self.loader, self.dataset = loader, loader.dataset
        self.expected = identity_rows(frame)
        self.paths = frame[["image_path", "rendered_mask_path"]].to_dict("records")

    def __iter__(self):
        position = 0
        for batch in self.loader:
            for i in range(len(batch["image"])):
                row = {field: batch[field][i] for field in ID_FIELDS}
                paths = {"image_path": batch["path"][i], "rendered_mask_path": batch["rendered_mask_path"][i]}
                if (position >= len(self.expected) or row != self.expected[position]
                    or paths != self.paths[position] or CLASS_ORDER[int(batch["label"][i])] != row["true_class"]):
                    raise ValueError("Evaluation loader changed exact sample identities/labels/paths")
                position += 1
            yield batch
        if position != len(self.expected):
            raise ValueError("Evaluation loader omitted frozen samples")


def execute_ablation(prepared, layout, *, device):
    from . import phase4_ablation_runner as boundary

    bundle, lock = boundary.frozen_bundle()
    validate_identity(prepared.identity, prepared.config, bundle)
    manifest, ready, supervision = boundary.revalidate_inputs(prepared.config, lock)
    if ready.to_dict() != prepared.readiness or manifest.to_dict() != prepared.manifest:
        raise ValueError("Ablation readiness changed before data/model construction")
    boundary.assert_supervision(prepared.supervision, prepared.identity["inputs"])
    boundary.guard_image_paths(prepared.supervision)
    boundary.recheck_prepared(prepared)
    if layout.experiment_dir != prepared.output_dir or any(layout.checkpoints_dir.iterdir()):
        raise ValueError("Ablation requires its fresh independent checkpoint directory")
    if device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError("Exactly one explicitly isolated CUDA device must be visible")
    if device not in ("cpu", "cuda"):
        raise ValueError("Unsupported explicit ablation device")
    config, identity = prepared.config, prepared.identity
    controls = config["determinism"]
    rng = set_global_seed(prepared.spec.seed, deterministic_algorithms=controls["deterministic_algorithms"],
                          cudnn_benchmark=controls["cudnn_benchmark"], warn_only=controls["warn_only"], require_torch=True)
    rendered = boundary.ROOT / config["data"]["rendered_dir"]
    # This existing helper only creates train/val datasets. Its final split policy
    # does not confer any final-run identity on this ablation.
    frames, datasets = build_phase4_initial_datasets(prepared.supervision, config, rendered_root=rendered)
    train = _loader(datasets["train"], config, seed=prepared.spec.seed, shuffle=True)
    val = ExactEvalLoader(_loader(datasets["val"], config, seed=prepared.spec.seed, shuffle=False), frames["val"])
    weights = compute_training_class_weights(prepared.supervision)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    model = _model(config).to(device)  # Fresh seeded construction; no reference weights loaded.
    events = ["fresh_model_initialized"]
    settings = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=float(settings["scheduler_factor"]),
                                                          patience=int(settings["scheduler_patience"]))
    common = {"device": device, "optimizer": optimizer, "class_weights": class_weights,
              "lambda_ordinal": identity["lambda_ordinal"], "lambda_attention": identity["lambda_attention"],
              "ablation": config["ablation"], "grad_clip_norm": settings["grad_clip_norm"]}
    best_loss, best_epoch, stale = math.inf, 0, 0
    history = []
    for epoch in range(1, int(settings["epochs"]) + 1):
        train_metrics, _ = _run_epoch(model, train, training=True, **common)
        validation, _ = _run_epoch(model, val, training=False, **common)
        loss = float(validation["total_loss"])
        if not math.isfinite(loss):
            raise ValueError("Ablation validation total loss is nonfinite")
        scheduler.step(loss)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation})
        improved = loss < best_loss
        if improved:
            best_loss, best_epoch, stale = loss, epoch, 0
        else:
            stale += 1
        for kind in (("best", "last") if improved else ("last",)):
            record = checkpoint_payload(identity, kind=kind, epoch=epoch, best_epoch=best_epoch, best_loss=best_loss,
                                        validation=validation, model=model, optimizer=optimizer, scheduler=scheduler)
            save_checkpoint(record, layout.checkpoints_dir / f"{kind}.pt")
        if stale >= int(settings["early_stopping_patience"]):
            break
    best, last = load_checkpoint_pair(layout.checkpoints_dir, identity, history, device=device)
    events.extend(["best_selected_from_validation", "persisted_best_last_validated"])
    model.load_state_dict(best["model_state"], strict=True)
    events.append("best_weights_reloaded")
    val_metrics, val_logits = _run_epoch(model, val, training=False, **common)
    predictions = {"val": build_predictions(frames["val"], val_logits, identity)}
    metrics = {"val": val_metrics}
    events.append("validation_predictions_exported")
    # First test dataset/loader construction occurs only after persisted best reload.
    test_dataset = _build_test_dataset(frames["test"], config, rendered)
    test = ExactEvalLoader(_loader(test_dataset, config, seed=prepared.spec.seed, shuffle=False), frames["test"])
    events.append("test_loader_created_after_best_reload")
    metrics["test"], test_logits = _run_epoch(model, test, training=False, **common)
    predictions["test"] = build_predictions(frames["test"], test_logits, identity)
    return {"history": history, "best_record": best, "last_record": last, "predictions": predictions,
            "metrics": metrics, "events": events, "class_weights": weights.tolist(), "seed_controls": rng}
