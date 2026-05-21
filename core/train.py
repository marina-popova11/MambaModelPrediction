import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm

def run_training_classification(model, train_loader, val_loader, epochs=20,
                                learning_rate=3e-4, early_stop_patience=10,
                                model_path="best_mamba_classifier.pth",
                                class_weights=None):
    device = next(model.parameters()).device
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    print("\nStarting Training(Classification)...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)  # [batch, n_classes]
            loss = criterion(logits, yb)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"DETECTED NaN/Inf Loss! Stopping.")
                return None

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            train_total += yb.size(0)
            train_correct += (predicted == yb).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                val_total += yb.size(0)
                val_correct += (predicted == yb).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'epoch': epoch
            }, model_path)
            print(f"-> Saved new best model! (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"\nTraining finished. Best Val Acc: {best_val_acc:.4f}, Best Val Loss: {best_val_loss:.4f}")
    return history