import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to run safely without GUI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def compute_metrics(eval_pred):
    """
    Computes accuracy and macro F1 score from prediction logits.
    The key returned MUST be 'f1_macro' to match metric_for_best_model.
    
    Parameters:
        eval_pred (tuple): A tuple containing (predictions, label_ids).
        
    Returns:
        dict: A dictionary containing accuracy and f1_macro metrics.
    """
    predictions, labels = eval_pred
    # Apply argmax to get the predicted class index for each sample
    preds = np.argmax(predictions, axis=-1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    
    return {
        "accuracy": acc,
        "f1_macro": f1
    }

def detailed_classification_report(y_true, y_pred, label_names):
    """
    Generates a detailed text classification report using scikit-learn.
    
    Parameters:
        y_true (list/array): Ground truth labels.
        y_pred (list/array): Predicted labels.
        label_names (list of str): Human-readable class names.
        
    Returns:
        str: Text classification report.
    """
    return classification_report(y_true, y_pred, target_names=label_names)

def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    """
    Generates a confusion matrix heatmap and saves it to a file.
    Does not display the plot (plt.show) to avoid GUI/headless errors.
    
    Parameters:
        y_true (list/array): Ground truth labels.
        y_pred (list/array): Predicted labels.
        label_names (list of str): Names of the class labels.
        save_path (str): Filepath where the plot image should be saved.
        
    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names, cmap='Blues')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_training_curves(log_history, save_path):
    """
    Parses and plots the loss and validation F1 macro progression from Trainer's log history.
    Saves the plot to save_path.
    
    Parameters:
        log_history (list of dict): The log_history attribute from trainer.state.
        save_path (str): Filepath where the curves plot image should be saved.
        
    Returns:
        None
    """
    train_epochs = []
    train_losses = []
    val_epochs = []
    val_losses = []
    val_f1s = []
    
    for entry in log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        
        # Round the epoch to avoid floating point representation precision issues
        epoch = round(epoch, 4)
        
        if "loss" in entry:
            train_epochs.append(epoch)
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            val_epochs.append(epoch)
            val_losses.append(entry["eval_loss"])
            val_f1s.append(entry.get("eval_f1_macro", 0.0))
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Plot Loss Curves
    if train_epochs:
        ax1.plot(train_epochs, train_losses, label="Train Loss", color="royalblue", marker='o', linestyle='-')
    if val_epochs:
        ax1.plot(val_epochs, val_losses, label="Val Loss", color="crimson", marker='s', linestyle='--')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Plot F1 Macro Curve
    if val_epochs:
        ax2.plot(val_epochs, val_f1s, label="Val F1 Macro", color="forestgreen", marker='^', linestyle='-')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Macro")
    ax2.set_title("Validation F1 Macro")
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()