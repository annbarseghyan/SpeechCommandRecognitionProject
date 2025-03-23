from sklearn.metrics import roc_curve, auc, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

def validate_and_evaluate(model, val_loader, criterion, device, root_dir, results_path, cnn=True):
    model.eval()  
    val_loss = 0
    correct_val = 0
    total_val = 0
    all_labels = []
    all_preds = []
    all_probs = []  

    
    class_names = sorted(os.listdir(root_dir))

    
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with torch.no_grad():
        for mfccs, labels in val_loader:
            mfccs = mfccs.to(torch.float32).to(device)
            labels = labels.to(device)

            if cnn:
                mfccs = mfccs.unsqueeze(1)
            else:
                mfccs = torch.flatten(mfccs, start_dim=1)

            outputs = model(mfccs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())  # Save probabilities

    
    all_probs = np.array(all_probs)

    
    all_labels_bin = label_binarize(all_labels, classes=np.unique(all_labels))

    
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(all_probs.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    val_accuracy = 100 * correct_val / total_val
    avg_val_loss = val_loss / len(val_loader)
    
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(results_path, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.show()
    plt.close()

    
    plt.figure(figsize=(6,6))
    for i in range(all_probs.shape[1]):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve for {class_names[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_curve_path = os.path.join(results_path, 'roc_curve.png')
    plt.savefig(roc_curve_path)
    plt.show()
    plt.close()

    return avg_val_loss, val_accuracy, precision, recall, cm, fpr, tpr, roc_auc
