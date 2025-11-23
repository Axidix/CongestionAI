import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


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
    early_stopping=None     
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

    scheduler_is_iter_based = (
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

            # -------------------------
            # ITER-BASED LR SCHEDULING
            # -------------------------
            if scheduler_is_iter_based:
                scheduler.step()  # called every batch

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

        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- Train Loss: {epoch_train_loss:.4f} "
            f"- Val Loss: {epoch_val_loss:.4f}"
        )

        # -------------------------
        # EPOCH-BASED SCHEDULERS
        # -------------------------
        if scheduler_is_plateau:
            scheduler.step(epoch_val_loss)

        # -------------------------
        # Early Stopping
        # -------------------------
        if early_stopping is not None:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_state = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                patience = early_stopping.get("patience", 5)

                if no_improve >= patience:
                    tqdm.write("Early stopping triggered.")
                    model.load_state_dict(best_state)
                    return train_losses, val_losses, best_state

    if best_state is None:
        best_state = model.state_dict()

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
