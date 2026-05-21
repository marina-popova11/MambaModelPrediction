import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

def evaluate_classification(model, test_loader, class_medians, class_labels,
                            time_index=None, model_path="best_mamba_classifier.pth"):
    print("Keys in class_medians:", list(class_medians.keys()))
    print("Expected class indices:", list(range(len(class_labels))))
    assert set(class_medians.keys()) == set(range(len(class_labels))), "Mismatch!"
    device = next(model.parameters()).device

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nEvaluating on Test Set... (Best Val Acc: {checkpoint['best_val_acc']:.4f})")

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)  # [batch, n_classes]
            preds = torch.argmax(logits, dim=1)  # [batch]

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_targets, all_preds)

    print(f"\n=== CLASSIFICATION RESULTS ===")
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"Weighted F1-Score: {f1:.4f}")
    print(f"\nClassification Report:")
    report = classification_report(all_targets, all_preds,
                                   target_names=class_labels,
                                   zero_division=0)
    print(report)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()

    if class_medians:
        pred_log_returns = np.array([class_medians[p] for p in all_preds])
        true_log_returns = np.array([class_medians[t] for t in all_targets])
        rmse_log = np.sqrt(np.mean((pred_log_returns - true_log_returns) ** 2))
        mae_log = np.mean(np.abs(pred_log_returns - true_log_returns))

        print(f"\n=== PRICE SPACE METRICS (using class medians) ===")
        print(f"RMSE (log-return): {rmse_log:.6f}")
        print(f"MAE (log-return): {mae_log:.6f}")
        print(f"Это соответствует ошибке в цене ~ {mae_log * 100:.2f}%")

        pred_direction = np.sign(pred_log_returns)
        true_direction = np.sign(true_log_returns)
        dir_accuracy = np.mean(pred_direction == true_direction)
        print(f"Direction Accuracy: {dir_accuracy:.4f} ({dir_accuracy:.2%})")

        if time_index is not None and len(time_index) == len(all_targets):
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))

            #Сравнение классов
            axes[0].plot(time_index, all_targets, label='Actual Class', color='blue', alpha=0.6)
            axes[0].plot(time_index, all_preds, label='Predicted Class', color='red', alpha=0.6, linestyle='--')
            axes[0].set_ylabel('Class Index')
            axes[0].set_title(f'Class Predictions (Accuracy: {accuracy:.2%})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Log-returns (через медианы)
            axes[1].plot(time_index, true_log_returns, label='Actual Log-Return', color='blue', alpha=0.7)
            axes[1].plot(time_index, pred_log_returns, label='Predicted Log-Return', color='red', alpha=0.7, linestyle='--')
            axes[1].set_ylabel('Log-Return')
            axes[1].set_title(f'Log-Returns from Class Medians (Dir Acc: {dir_accuracy:.2%})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('classification_results.png', dpi=150)
            plt.show()

    return accuracy, f1