"""
BioHAN: Biologically-Informed Hierarchical Attention Networks
Main model implementation with biological constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from .biological_knowledge import BiologicalKnowledgeBase


class GeneRegulatoryModule(nn.Module):
    """Gene regulatory module discovery layer

    Discovers functional gene modules based on expression patterns
    and biological prior knowledge.
    """

    def __init__(self, gene_list: List[str], n_modules: int = 50):
        super().__init__()
        self.n_genes = len(gene_list)
        self.n_modules = min(n_modules, max(10, self.n_genes // 20))

        # Initialize with biological priors
        kb = BiologicalKnowledgeBase()
        prior_assignment = kb.get_module_prior(gene_list, self.n_modules)

        self.module_assignment = nn.Parameter(torch.FloatTensor(prior_assignment.T))
        self.module_interaction = nn.Parameter(torch.eye(self.n_modules))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through regulatory modules

        Args:
            x: Input gene expression tensor [batch_size, n_genes]

        Returns:
            Dictionary containing:
                - activity: Module activity levels [batch_size, n_modules]
                - weights: Module assignment weights [n_modules, n_genes]
                - sparsity_loss: L1 regularization term
        """
        # Soft module assignment with softmax
        module_weights = F.softmax(self.module_assignment, dim=1)

        # Calculate module activity
        module_activity = torch.matmul(x, module_weights.T)

        # Module-module interactions
        regulated_activity = torch.matmul(module_activity, torch.sigmoid(self.module_interaction))

        return {
            'activity': regulated_activity,
            'weights': module_weights,
            'sparsity_loss': torch.mean(torch.abs(module_weights)) * 0.01
        }


class BiologicalAttention(nn.Module):
    """Biologically-constrained multi-head attention mechanism

    Incorporates protein-protein interaction networks to guide
    attention weights toward biologically meaningful relationships.
    """

    def __init__(self, hidden_dim: int, gene_list: List[str], n_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        # Get gene interaction matrix from biological databases
        kb = BiologicalKnowledgeBase()
        interaction_matrix = kb.get_interaction_matrix(gene_list)
        self.register_buffer('interaction_mask', torch.FloatTensor(interaction_matrix))

        # Attention parameters
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply biologically-constrained attention

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Output tensor with same shape as input
        """
        B, N, C = x.shape

        # Generate Q, K, V matrices
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Apply biological constraints
        if hasattr(self, 'interaction_mask') and N == len(self.interaction_mask):
            bio_mask = self.interaction_mask.unsqueeze(0).unsqueeze(0)
            # Modulate attention based on known interactions
            attn = attn * (0.1 + 0.9 * bio_mask)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


class TemperatureScaledSigmoid(nn.Module):
    """Temperature-scaled sigmoid activation for sharp marker selection"""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x / self.temperature)


class BioHAN(nn.Module):
    """Biologically-Informed Hierarchical Attention Network

    Main model combining biological constraints, hierarchical attention,
    and interpretable marker gene discovery for cell type classification.

    Args:
        gene_list: List of gene names in expression data
        hidden_dim: Hidden dimension size (default: 256)
        n_classes: Number of cell types to classify
        n_modules: Number of gene regulatory modules (default: 100)
        n_layers: Number of attention layers (default: 3)
        dropout: Dropout rate (default: 0.2)
    """

    def __init__(self,
                 gene_list: List[str],
                 hidden_dim: int = 256,
                 n_classes: int = 50,
                 n_modules: int = 100,
                 n_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()

        self.gene_list = gene_list
        self.n_genes = len(gene_list)
        self.n_classes = n_classes

        # Biological knowledge base
        self.kb = BiologicalKnowledgeBase()

        # 1. Gene expression encoder with layer normalization
        self.gene_encoder = nn.Sequential(
            nn.Linear(self.n_genes, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 2. Gene regulatory module discovery
        self.regulatory_modules = GeneRegulatoryModule(gene_list, n_modules)

        # 3. Biologically-constrained attention layers
        self.attention_layers = nn.ModuleList([
            BiologicalAttention(hidden_dim, gene_list)
            for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        # 4. Hierarchical cell state encoding
        self.n_states = 32
        self.state_prototypes = nn.Parameter(torch.randn(self.n_states, hidden_dim))
        self.state_temperature = 0.5

        # 5. Cell type classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

        # 6. Temperature-scaled marker gene learning
        self.marker_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, self.n_genes),
                TemperatureScaledSigmoid(temperature=0.1)
            ) for _ in range(n_classes)
        ])

        # Initialize marker weight storage
        self.register_buffer('learned_marker_weights', torch.zeros(n_classes, self.n_genes))

        # 7. Initialize with biological priors
        self._initialize_with_biological_priors()

    def _initialize_with_biological_priors(self):
        """Initialize marker attention weights using known cell type markers"""
        celltype_markers = self.kb.get_celltype_markers()

        for class_idx in range(min(self.n_classes, len(celltype_markers))):
            # Collect possible marker genes for this class
            markers = []
            for celltype, marker_genes in celltype_markers.items():
                markers.extend(marker_genes)

            # Boost initial weights for known markers
            with torch.no_grad():
                linear_layer = self.marker_attention[class_idx][3]  # Last linear layer
                for gene in markers:
                    if gene in self.gene_list:
                        gene_idx = self.gene_list.index(gene)
                        if gene_idx < linear_layer.weight.size(1):
                            linear_layer.weight[:, gene_idx] *= 2.0

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass through BioHAN

        Args:
            x: Gene expression input [batch_size, n_genes]
            return_features: If True, return only features without classification

        Returns:
            Dictionary containing:
                - logits: Classification logits [batch_size, n_classes]
                - features: Cell state features [batch_size, hidden_dim]
                - regulatory_loss: Sparsity loss from regulatory modules
                - marker_scores: Marker gene importance scores
                - marker_importance: Normalized marker importance per class
        """
        batch_size = x.size(0)

        # 1. Encode gene expression
        gene_features = self.gene_encoder(x)

        # 2. Process through regulatory modules
        regulatory_output = self.regulatory_modules(x)

        # 3. Apply hierarchical attention
        features = gene_features.unsqueeze(1)  # Add sequence dimension
        for attn, ln in zip(self.attention_layers, self.layer_norms):
            residual = features
            features = attn(features)
            features = ln(features + residual)

        features = features.squeeze(1)  # Remove sequence dimension

        # 4. Encode into hierarchical cell states
        state_similarities = F.cosine_similarity(
            features.unsqueeze(1),
            self.state_prototypes.unsqueeze(0),
            dim=2
        )
        state_weights = F.softmax(state_similarities / self.state_temperature, dim=1)
        cell_states = torch.matmul(state_weights, self.state_prototypes)

        if return_features:
            return cell_states

        # 5. Classify cell types
        logits = self.classifier(cell_states)

        # 6. Learn marker genes
        marker_scores = []
        if self.training:
            # During training: compute marker scores dynamically
            for class_idx in range(self.n_classes):
                class_marker_scores = self.marker_attention[class_idx](cell_states)
                marker_scores.append(class_marker_scores)

            marker_scores = torch.stack(marker_scores, dim=1)  # [batch, n_classes, n_genes]

            # Update learned marker weights using exponential moving average
            with torch.no_grad():
                pred_classes = logits.argmax(dim=1)
                for i in range(batch_size):
                    cls = pred_classes[i]
                    # Combine attention scores with expression magnitude
                    importance = marker_scores[i, cls] * torch.abs(x[i])
                    # Smooth update
                    self.learned_marker_weights[cls] = (
                            0.98 * self.learned_marker_weights[cls] +
                            0.02 * importance
                    )
        else:
            # During inference: use learned weights
            marker_scores = self.learned_marker_weights.unsqueeze(0).expand(batch_size, -1, -1)

        return {
            'logits': logits,
            'features': cell_states,
            'regulatory_loss': regulatory_output['sparsity_loss'],
            'marker_scores': marker_scores,
            'marker_importance': F.normalize(self.learned_marker_weights, p=1, dim=1)
        }

    def get_interpretations(self, x: torch.Tensor) -> Dict[str, any]:
        """Get biological interpretations for predictions

        Args:
            x: Gene expression input [batch_size, n_genes]

        Returns:
            Dictionary containing:
                - top_markers: Top marker genes per cell type
                - predictions: Predicted cell type indices
        """
        with torch.no_grad():
            outputs = self(x)

            # Extract top marker genes per class
            marker_importance = outputs['marker_importance'].cpu().numpy()
            top_markers = {}

            for class_idx in range(self.n_classes):
                importance = marker_importance[class_idx]
                # Get top 20 most important genes
                top_indices = np.argsort(importance)[-20:][::-1]
                top_markers[f'class_{class_idx}'] = [
                    (self.gene_list[idx], float(importance[idx]))
                    for idx in top_indices
                    if importance[idx] > 0.0001
                ]

            return {
                'top_markers': top_markers,
                'predictions': outputs['logits'].argmax(dim=1).cpu().numpy()
            }