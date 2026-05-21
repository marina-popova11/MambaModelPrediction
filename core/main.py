# acc 22, dir acc 56
import torch
import numpy as np
import torch.nn as nn
from data import get_data_loaders
from model import *
from train import run_training_classification
from evaluate import evaluate_classification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    LOOKBACK = 120
    BATCH_SIZE = 128
    EPOCHS = 20
    D_MODEL = 128
    N_LAYERS = 2
    N_BINS = 5
    CSV_FILE = 'btc_1h_2020-2025_training_sample_with_bins_second.csv'
    CACHE_FILE = 'processed_data_cache.pkl'
    MODEL_PATH = 'best_mamba_classifier.pth'

    data_dict, train_loader, val_loader, test_loader = get_data_loaders(
        filename=CSV_FILE,
        lookback=LOOKBACK,
        batch_size=BATCH_SIZE,
        cache_file=CACHE_FILE,
        n_bins=N_BINS,
        classification_mode=True,
        force_reprocess=True
    )

    input_dim = data_dict['X_train'].shape[2]
    n_classes = data_dict['n_classes']
    class_labels = data_dict['class_labels']
    class_medians = data_dict['class_medians']

    from sklearn.utils.class_weight import compute_class_weight
    class_weights_np = compute_class_weight('balanced', classes=np.unique(data_dict['y_train']), y=data_dict['y_train'])
    class_weights_tensor = torch.FloatTensor(class_weights_np)
    print(f"Class weights: {class_weights_tensor}")

    model = MambaModel(
        input_dim=input_dim,
        n_classes=n_classes,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        use_mamba=True
    ).to(device)

    history = run_training_classification(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        learning_rate=3e-4,
        early_stop_patience=10,
        model_path=MODEL_PATH,
        class_weights=class_weights_tensor
    )

    if history is not None:
        evaluate_classification(
            model=model,
            test_loader=test_loader,
            class_medians=class_medians,
            class_labels=class_labels,
            time_index=data_dict['time_index']['test'],
            model_path=MODEL_PATH
        )