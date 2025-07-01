"""
Enhanced evaluation metrics for comprehensive model assessment
"""

import numpy as np
import torch
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    accuracy_score, silhouette_score, f1_score,
    precision_recall_fscore_support, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef
)
from typing import Dict, Optional, Union, List, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def evaluate_model(model: torch.nn.Module,
                   test_loader: torch.utils.data.DataLoader,
                   device: Union[str, torch.device],
                   feature_key: str = 'features',
                   logit_key: str = 'logits',
                   comprehensive: bool = False) -> Dict[str, float]:
    """Evaluate model performance with multiple metrics

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        feature_key: Key for features in model output dict
        logit_key: Key for logits in model output dict
        comprehensive: Whether to compute all metrics

    Returns:
        Dictionary of evaluation metrics
    """
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()
    model = model.to(device)

    all_features = []
    all_predictions = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)

            # Get model outputs
            outputs = model(batch_x)

            # Handle different output formats
            if isinstance(outputs, dict):
                if feature_key in outputs:
                    features = outputs[feature_key]
                else:
                    features = batch_x  # Use raw features as fallback
                logits = outputs.get(logit_key, outputs.get('logits'))
            else:
                logits = outputs
                features = batch_x

            predictions = logits.argmax(dim=1)

            all_features.append(features.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_logits.append(logits.cpu().numpy())

    # Combine results
    features = np.vstack(all_features)
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    logits = np.vstack(all_logits)

    # Core metrics
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'ari': adjusted_rand_score(labels, predictions),
        'nmi': normalized_mutual_info_score(labels, predictions)
    }

    # Additional metrics
    if comprehensive:
        # F1 scores
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')

        # Precision, Recall, F1 per class
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )

        # Other metrics
        metrics.update({
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'kappa': cohen_kappa_score(labels, predictions),
            'mcc': matthews_corrcoef(labels, predictions),
            'precision_mean': np.mean(precision),
            'recall_mean': np.mean(recall),
            'n_classes_predicted': len(np.unique(predictions))
        })

        # Per-class metrics
        metrics['per_class'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        }

    # Silhouette score (if feature dimension is reasonable)
    if features.shape[1] < 1000 and len(np.unique(labels)) > 1:
        try:
            metrics['silhouette'] = silhouette_score(features, labels, sample_size=min(5000, len(labels)))
        except:
            metrics['silhouette'] = np.nan

    return metrics


def calculate_marker_consistency(model1_markers: Dict[int, List[str]],
                                 model2_markers: Dict[int, List[str]],
                                 top_k: int = 50) -> float:
    """Calculate marker gene consistency between two models

    Args:
        model1_markers: Marker genes from model 1
        model2_markers: Marker genes from model 2
        top_k: Number of top markers to consider

    Returns:
        Average Jaccard similarity across cell types
    """
    similarities = []

    for class_idx in set(model1_markers.keys()) & set(model2_markers.keys()):
        set1 = set(model1_markers[class_idx][:top_k])
        set2 = set(model2_markers[class_idx][:top_k])

        if len(set1) == 0 and len(set2) == 0:
            similarity = 1.0
        elif len(set1) == 0 or len(set2) == 0:
            similarity = 0.0
        else:
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            similarity = intersection / union if union > 0 else 0.0

        similarities.append(similarity)

    return np.mean(similarities) if similarities else 0.0


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          save_path: Optional[Path] = None,
                          normalize: bool = True) -> plt.Figure:
    """Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save figure
        normalize: Whether to normalize the matrix

    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def calculate_classification_report(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    class_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Generate detailed classification report

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes

    Returns:
        DataFrame with classification metrics per class
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    # Create DataFrame
    report_data = {
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'support': support
    }

    if class_names is not None:
        df = pd.DataFrame(report_data, index=class_names)
    else:
        df = pd.DataFrame(report_data)

    # Add averages
    avg_data = {
        'precision': [
            precision_recall_fscore_support(y_true, y_pred, average='macro')[0],
            precision_recall_fscore_support(y_true, y_pred, average='weighted')[0]
        ],
        'recall': [
            precision_recall_fscore_support(y_true, y_pred, average='macro')[1],
            precision_recall_fscore_support(y_true, y_pred, average='weighted')[1]
        ],
        'f1-score': [
            f1_score(y_true, y_pred, average='macro'),
            f1_score(y_true, y_pred, average='weighted')
        ],
        'support': [len(y_true), len(y_true)]
    }

    avg_df = pd.DataFrame(avg_data, index=['macro avg', 'weighted avg'])

    # Combine
    df = pd.concat([df, avg_df])

    return df


def evaluate_marker_quality(markers: Dict[str, List[Tuple[str, float]]],
                            known_markers: Dict[str, List[str]]) -> Dict[str, float]:
    """Evaluate quality of discovered markers against known markers

    Args:
        markers: Discovered markers with importance scores
        known_markers: Known marker genes per cell type

    Returns:
        Dictionary of evaluation metrics
    """
    results = {}

    precisions = []
    recalls = []

    for cell_type, discovered in markers.items():
        if cell_type in known_markers:
            known = set(known_markers[cell_type])
            discovered_genes = set([gene for gene, _ in discovered[:20]])  # Top 20

            if len(discovered_genes) > 0:
                precision = len(known & discovered_genes) / len(discovered_genes)
                recall = len(known & discovered_genes) / len(known) if len(known) > 0 else 0

                precisions.append(precision)
                recalls.append(recall)

    results['marker_precision'] = np.mean(precisions) if precisions else 0
    results['marker_recall'] = np.mean(recalls) if recalls else 0
    results['marker_f1'] = 2 * results['marker_precision'] * results['marker_recall'] / \
                           (results['marker_precision'] + results['marker_recall']) \
        if (results['marker_precision'] + results['marker_recall']) > 0 else 0

    return results


def evaluate_batch_effect(features: np.ndarray,
                          labels: np.ndarray,
                          batch_labels: np.ndarray) -> Dict[str, float]:
    """Evaluate batch effect in learned representations

    Args:
        features: Learned feature representations
        labels: Cell type labels
        batch_labels: Batch labels

    Returns:
        Dictionary of batch effect metrics
    """
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    # Reduce dimensions if needed
    if features.shape[1] > 50:
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)

    # Cell type clustering quality (should be high)
    cell_type_silhouette = silhouette_score(features, labels, sample_size=min(5000, len(labels)))

    # Batch clustering quality (should be low)
    if len(np.unique(batch_labels)) > 1:
        batch_silhouette = silhouette_score(features, batch_labels, sample_size=min(5000, len(labels)))
    else:
        batch_silhouette = 0

    # Batch effect score (lower is better)
    batch_effect_score = batch_silhouette / (cell_type_silhouette + 1e-6)

    return {
        'cell_type_silhouette': cell_type_silhouette,
        'batch_silhouette': batch_silhouette,
        'batch_effect_score': batch_effect_score
    }