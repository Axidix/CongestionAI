import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.sequences import NHitsDataset
import numpy as np
import copy
import time
import os
from typing import Tuple, List
from dataclasses import asdict
from torch import nn
from torch.amp import GradScaler

from src.utils.configs import TCNModelConfig, TransformerModelConfig, TrainingConfig, DataConfig
from src.models.tcn_forecaster import MultiHeadTCNForecaster
from src.models.transformer_forecaster import MultiHeadTransformerForecaster
from src.model_pipelines.losses import LossConfig, create_loss
from src.utils.plots import plot_training_curves
from src.utils.memmap_sequences import MemmapDataset

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,         
    scaler=None,
    device="cuda",
    num_epochs=50,
    grad_clip=1.0,   
    patience=None     
):

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    # detect scheduler style
    scheduler_is_plateau = (
        scheduler is not None and 
        "plateau" in scheduler.__class__.__name__.lower()
    )

    # CosineAnnealingLR should be stepped per epoch, not per iteration
    scheduler_is_epoch_based = (
        scheduler is not None and 
        not scheduler_is_plateau
    )

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0

        # -------------------------
        # TRAINING LOOP
        # -------------------------
        for X_batch, Y_batch, det_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - training"):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            det_ids = det_ids.to(device)

            optimizer.zero_grad()

            # AMP branch
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    preds = model(X_batch, det_ids)
                    loss = criterion(preds, Y_batch)

                scaler.scale(loss).backward()

                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()

            # Non-AMP branch
            else:
                preds = model(X_batch, det_ids)
                loss = criterion(preds, Y_batch)
                loss.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # -------------------------
        # VALIDATION LOOP
        # -------------------------
        model.eval()
        epoch_val_loss = 0

        with torch.no_grad():
            for X_batch, Y_batch, det_ids in tqdm(val_loader, desc="Validating"):
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                det_ids = det_ids.to(device)

                preds = model(X_batch, det_ids)
                loss = criterion(preds, Y_batch)
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

        # -------------------------
        # EPOCH-BASED SCHEDULERS (CosineAnnealingLR, StepLR, etc.)
        # -------------------------
        if scheduler_is_epoch_based:
            scheduler.step()
        elif scheduler_is_plateau:
            scheduler.step(epoch_val_loss)

        # -------------------------
        # Early Stopping & Best Model Tracking
        # -------------------------
        # FIX: Check if val_loss is LESS than best (improvement)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Deep copy to avoid reference issues
            if hasattr(model, '_orig_mod'):
                best_state = copy.deepcopy(model._orig_mod.state_dict())
            else:
                best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            tqdm.write(
                f"Epoch {epoch+1}/{num_epochs} "
                f"- Train Loss: {epoch_train_loss:.4f} "
                f"- Val Loss: {epoch_val_loss:.4f} ✓ New best!"
            )
        else:
            no_improve += 1
            tqdm.write(
                f"Epoch {epoch+1}/{num_epochs} "
                f"- Train Loss: {epoch_train_loss:.4f} "
                f"- Val Loss: {epoch_val_loss:.4f} "
                f"(no improve: {no_improve}/{patience}, best: {best_val_loss:.4f})"
            )
            
            if patience is not None and no_improve >= patience:
                tqdm.write("Early stopping triggered.")
                break

    # Load best state back into model before returning
    if best_state is not None:
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)
    else:
        # Safety: if no best_state, use current state
        if hasattr(model, '_orig_mod'):
            best_state = copy.deepcopy(model._orig_mod.state_dict())
        else:
            best_state = copy.deepcopy(model.state_dict())

    return train_losses, val_losses, best_state


def evaluate(model, loader, criterion, device="cuda"):
    model.eval()
    total_loss = 0
    preds_list = []

    with torch.no_grad():
        for X_batch, Y_batch, det_ids in tqdm(loader, desc="Evaluating"):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            det_ids = det_ids.to(device)

            preds = model(X_batch, det_ids)
            loss = criterion(preds, Y_batch)

            total_loss += loss.item()
            preds_list.append(preds.cpu())

    preds_list = torch.cat(preds_list, dim=0)
    return preds_list, total_loss / len(loader)


def predict(model, X_hist, det_ids, device="cuda", batch_size=256):
    """
    Run inference on arbitrary (X_hist, det_ids) pairs.
    Returns an array of predictions (N, horizon).
    """

    model.eval()
    preds_list = []

    loader = DataLoader(
        NHitsDataset(X_hist, np.zeros((len(X_hist), 1), dtype=np.float32), det_ids),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    with torch.no_grad():
        for X_batch, _, det_id_batch in loader:
            X_batch = X_batch.to(device)
            det_id_batch = det_id_batch.to(device)
            preds = model(X_batch, det_id_batch)
            preds_list.append(preds.cpu())

    return torch.cat(preds_list, dim=0)



def create_model(
    model_cfg,
    num_features: int,
    num_detectors: int,
    forecast_horizon: int,
    device: str = "cuda"
) -> nn.Module:
    """Create model based on configuration."""
    
    if isinstance(model_cfg, TCNModelConfig):
        print(f"\nCreating TCN model...")
        model = MultiHeadTCNForecaster(
            num_features=num_features,
            horizon=forecast_horizon,
            num_detectors=num_detectors,
            emb_dim=model_cfg.emb_dim,
            num_channels=model_cfg.num_channels,
            kernel_size=model_cfg.kernel_size,
            dropout_encoder=model_cfg.dropout_encoder,
            dropout_heads=model_cfg.dropout_heads,
            use_se=model_cfg.use_se,
            pooling=model_cfg.pooling,
        )
    elif isinstance(model_cfg, TransformerModelConfig):
        print(f"\nCreating MultiHead Transformer model...")
        model = MultiHeadTransformerForecaster(
            num_features=num_features,
            horizon=forecast_horizon,
            num_detectors=num_detectors,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            num_layers=model_cfg.num_layers,
            dim_feedforward=model_cfg.dim_feedforward,
            dropout=model_cfg.dropout,
            det_emb_dim=model_cfg.det_emb_dim,
            pooling=model_cfg.pooling,
            max_seq_len=128
        )
    else:
        raise ValueError(f"Unknown model config type: {type(model_cfg)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model.to(device)


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_full_scale(
    model: nn.Module,
    X_train: np.memmap,
    Y_train: np.memmap,
    det_train: np.ndarray,
    X_val: np.memmap,
    Y_val: np.memmap,
    det_val: np.ndarray,
    train_cfg: TrainingConfig,
    model_cfg,
    data_cfg: DataConfig,
    exp_name: str,
    output_dir: str,
    device: str = "cuda",
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Full-scale training with memmap datasets.
    """
    print("\n" + "="*70)
    print(f"TRAINING: {exp_name}")
    print("="*70)
    
    # Create dataloaders with MemmapDataset
    train_loader = DataLoader(
        MemmapDataset(X_train, Y_train, det_train),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=train_cfg.num_workers,
        persistent_workers=True if train_cfg.num_workers > 0 else False,
        prefetch_factor=2 if train_cfg.num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        MemmapDataset(X_val, Y_val, det_val),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=train_cfg.num_workers,
        persistent_workers=True if train_cfg.num_workers > 0 else False,
    )
    
    print(f"\nDataloaders:")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches: {len(val_loader):,}")
    print(f"  Batch size: {train_cfg.batch_size}")
    print(f"  Num workers: {train_cfg.num_workers}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.betas,
    )
    print(f"\nOptimizer: AdamW(lr={train_cfg.lr}, weight_decay={train_cfg.weight_decay}, betas={train_cfg.betas})")
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.epochs,
        eta_min=1e-6
    )
    print(f"Scheduler: CosineAnnealingLR(T_max={train_cfg.epochs}, eta_min=1e-6)")
    
    # Create loss
    loss_cfg = LossConfig(
        loss_type="spike_weighted",
        spike_weight=train_cfg.spike_weight,
        spike_threshold=train_cfg.spike_threshold
    )
    criterion = create_loss(loss_cfg)
    print(f"Loss: SpikeWeightedMSE(weight={train_cfg.spike_weight}, threshold={train_cfg.spike_threshold})")
    
    # AMP scaler
    scaler = GradScaler('cuda') if train_cfg.use_amp else None
    print(f"AMP: {'enabled' if train_cfg.use_amp else 'disabled'}")
    print(f"Grad clip: {train_cfg.grad_clip}")
    
    # Train
    print(f"\nStarting training for {train_cfg.epochs} epochs...")
    start_time = time.time()
    
    train_losses, val_losses, best_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        num_epochs=train_cfg.epochs,
        grad_clip=train_cfg.grad_clip,
        patience=train_cfg.patience,
    )
    
    elapsed = time.time() - start_time
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    
    # Save losses to file
    with open(f"{output_dir}/losses_{exp_name}.txt", "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{i+1},{t_loss:.6f},{v_loss:.6f}\n")
    
    # Save training curves plot
    plot_training_curves(
        train_losses, val_losses,
        filename=f"training_curve_{exp_name}.png",
        dir=output_dir
    )
    
    # Prepare config dicts for saving
    model_cfg_dict = asdict(model_cfg)
    data_cfg_dict = asdict(data_cfg)
    train_cfg_dict = asdict(train_cfg)
    
    # Save BEST model checkpoint
    best_model_path = f"{output_dir}/checkpoints/best_model_{exp_name}.pt"
    torch.save({
        'model_state_dict': best_state,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': min(val_losses),
        'best_epoch': val_losses.index(min(val_losses)) + 1,
        'total_epochs': len(train_losses),
        'training_time_s': elapsed,
        'exp_name': exp_name,
        'model_config': model_cfg_dict,
        'data_config': data_cfg_dict,
        'training_config': train_cfg_dict,
        'num_features': X_train.shape[-1],
        'history_length': X_train.shape[1],
        'num_detectors': len(np.unique(det_train)),
    }, best_model_path)
    print(f"\n  Best model saved to: {best_model_path}")
    
    # Save FINAL model checkpoint (last epoch state)
    final_model_path = f"{output_dir}/checkpoints/final_model_{exp_name}.pt"
    
    # Get the underlying model if compiled
    if hasattr(model, '_orig_mod'):
        final_state = model._orig_mod.state_dict()
    else:
        final_state = model.state_dict()
    
    torch.save({
        'model_state_dict': final_state,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_val_loss': val_losses[-1],
        'total_epochs': len(train_losses),
        'training_time_s': elapsed,
        'exp_name': exp_name,
        'model_config': model_cfg_dict,
        'data_config': data_cfg_dict,
        'training_config': train_cfg_dict,
        'num_features': X_train.shape[-1],
        'history_length': X_train.shape[1],
        'num_detectors': len(np.unique(det_train)),
    }, final_model_path)
    print(f"  Final model saved to: {final_model_path}")
    
    print(f"\n✓ Training complete!")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Best val loss: {min(val_losses):.6f}")
    print(f"  Best epoch: {val_losses.index(min(val_losses)) + 1}")
    print(f"  Results saved to: {output_dir}/")
    
    # Load best state back into model
    if hasattr(model, '_orig_mod'):
        model._orig_mod.load_state_dict(best_state)
    else:
        model.load_state_dict(best_state)
    
    return model, train_losses, val_losses

