"""
Unified training logic with mixed precision and advanced optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Union, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                n_epochs: int = 50,
                learning_rate: float = 1e-3,
                weight_decay: float = 1e-5,
                device: Union[str, torch.device] = 'cuda',
                use_amp: bool = True,
                patience: int = 10,
                scheduler_type: str = 'plateau') -> Dict[str, List[float]]:
    """Unified training function with advanced features

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        test_loader: Validation data loader
        n_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        device: Device to train on
        use_amp: Whether to use automatic mixed precision
        patience: Early stopping patience
        scheduler_type: Type of learning rate scheduler

    Returns:
        Dictionary containing training history
    """

    # Setup device
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-6
        )
    else:
        scheduler = None

    # Mixed precision training
    scaler = GradScaler() if use_amp and torch.cuda.is_available() else None

    # Early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rates': []
    }

    # Best model tracking
    best_val_acc = 0
    best_model_state = None

    logger.info(f"Starting training on {device}")

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{n_epochs}')

        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            if scaler is not None:
                with autocast():
                    outputs = model(batch_x)
                    loss = compute_loss(outputs, batch_y, criterion, model)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x)
                loss = compute_loss(outputs, batch_y, criterion, model)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation phase
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        avg_train_loss = train_loss / train_steps

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Learning rate scheduling
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}, LR={current_lr:.6f}")

        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation accuracy: {best_val_acc:.3f}")

    return history


def compute_loss(outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                 labels: torch.Tensor,
                 criterion: nn.Module,
                 model: nn.Module) -> torch.Tensor:
    """Compute loss with support for different output formats

    Args:
        outputs: Model outputs (tensor or dictionary)
        labels: Ground truth labels
        criterion: Loss criterion
        model: Model instance (for accessing regularization terms)

    Returns:
        Total loss
    """
    # Handle different output formats
    if isinstance(outputs, dict):
        logits = outputs['logits']
        loss = criterion(logits, labels)

        # Add regularization losses
        if 'regulatory_loss' in outputs:
            loss = loss + outputs['regulatory_loss']

        # Marker gene learning losses
        if 'marker_scores' in outputs and model.training:
            marker_scores = outputs['marker_scores']

            # Sparsity loss: encourage focused marker selection
            sparsity_loss = torch.mean(torch.sum(marker_scores, dim=-1)) * 0.0001

            # Diversity loss: encourage different markers for different classes
            if marker_scores.size(0) > 1 and marker_scores.size(1) > 1:
                n_classes = marker_scores.size(1)

                # Normalize marker scores
                marker_probs = F.normalize(marker_scores, p=1, dim=-1)
                avg_marker_probs = marker_probs.mean(dim=0)

                # Calculate pairwise similarity
                similarity = torch.matmul(avg_marker_probs, avg_marker_probs.T)

                # Exclude diagonal
                mask = ~torch.eye(n_classes, device=marker_scores.device).bool()
                diversity_loss = torch.mean(similarity[mask]) * 0.1
            else:
                diversity_loss = 0

            # Entropy loss: encourage informative marker distributions
            entropy = -torch.sum(
                marker_scores * torch.log(marker_scores + 1e-8),
                dim=-1
            ).mean()
            entropy_loss = -entropy * 0.001

            loss = loss + sparsity_loss + diversity_loss + entropy_loss
    else:
        # Simple tensor output
        logits = outputs
        loss = criterion(logits, labels)

    return loss


def evaluate_model(model: nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float]:
    """Evaluate model performance

    Args:
        model: Model to evaluate
        dataloader: Data loader
        criterion: Loss criterion
        device: Device to evaluate on

    Returns:
        Average loss and accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)

            # Get logits
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            # Calculate loss
            loss = criterion(logits, batch_y)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


def train_with_cross_validation(model_class: type,
                                model_params: Dict,
                                data: Dict,
                                n_folds: int = 5,
                                n_epochs: int = 50,
                                device: str = 'cuda') -> Dict[str, List[float]]:
    """Train model with k-fold cross-validation

    Args:
        model_class: Model class to instantiate
        model_params: Parameters for model initialization
        data: Data dictionary from DataProcessor
        n_folds: Number of cross-validation folds
        n_epochs: Number of epochs per fold
        device: Device to train on

    Returns:
        Cross-validation results
    """
    from sklearn.model_selection import StratifiedKFold

    # Combine train and test data for CV
    X = np.vstack([data['X_train'], data['X_test']])
    y = np.hstack([data['y_train'], data['y_test']])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_results = {
        'fold_accuracies': [],
        'fold_losses': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Training fold {fold + 1}/{n_folds}")

        # Create fold data
        fold_data = {
            'X_train': X[train_idx],
            'X_test': X[val_idx],
            'y_train': y[train_idx],
            'y_test': y[val_idx],
            'gene_names': data['gene_names'],
            'n_classes': data['n_classes']
        }

        # Create data loaders
        from .data_processing import DataProcessor
        train_loader, val_loader = DataProcessor.create_dataloaders(fold_data)

        # Initialize model
        model = model_class(**model_params).to(device)

        # Train
        history = train_model(
            model, train_loader, val_loader,
            n_epochs=n_epochs, device=device
        )

        # Record results
        cv_results['fold_accuracies'].append(max(history['val_accuracy']))
        cv_results['fold_losses'].append(min(history['val_loss']))

    # Calculate statistics
    cv_results['mean_accuracy'] = np.mean(cv_results['fold_accuracies'])
    cv_results['std_accuracy'] = np.std(cv_results['fold_accuracies'])

    logger.info(f"Cross-validation complete: {cv_results['mean_accuracy']:.3f} "
                f"± {cv_results['std_accuracy']:.3f}")

    return cv_results