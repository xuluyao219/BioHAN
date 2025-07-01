"""
Cross-dataset consistency analysis for BioHAN
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from pathlib import Path
import json

from core.models import BioHAN
from core.data_processing import DataProcessor
from baselines.baseline_models import SimpleNN, VAEBaseline


class CrossDatasetConsistency:
    """Calculate cross-dataset marker gene consistency"""

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def extract_marker_genes(self,
                             model: torch.nn.Module,
                             data_loader: torch.utils.data.DataLoader,
                             gene_names: List[str],
                             n_classes: int,
                             top_k: int = 50) -> Dict[int, List[str]]:
        """Extract top-k marker genes for each cell type"""

        model.eval()

        if isinstance(model, BioHAN):
            # Use BioHAN's learned marker weights
            marker_importance = model.learned_marker_weights.cpu().numpy()

            markers_per_class = {}
            for class_idx in range(n_classes):
                importance = marker_importance[class_idx]
                top_indices = np.argsort(importance)[-top_k:][::-1]
                markers_per_class[class_idx] = [gene_names[idx] for idx in top_indices]

        else:
            # For other models, use gradient-based importance
            markers_per_class = self._gradient_based_markers(
                model, data_loader, gene_names, n_classes, top_k
            )

        return markers_per_class

    def _gradient_based_markers(self,
                                model: torch.nn.Module,
                                data_loader: torch.utils.data.DataLoader,
                                gene_names: List[str],
                                n_classes: int,
                                top_k: int) -> Dict[int, List[str]]:
        """Extract markers using gradient-based importance for baseline models"""

        # Initialize importance scores
        importance_scores = np.zeros((n_classes, len(gene_names)))
        class_counts = np.zeros(n_classes)

        model.eval()
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.device).requires_grad_(True)
            batch_y = batch_y.to(self.device)

            # Forward pass
            if isinstance(model, (SimpleNN, VAEBaseline)):
                if isinstance(model, VAEBaseline):
                    outputs = model(batch_x)
                    logits = outputs['logits']
                else:
                    logits = model(batch_x)
            else:
                outputs = model(batch_x)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            # Calculate gradients for each class
            for class_idx in range(n_classes):
                class_mask = (batch_y == class_idx)
                if class_mask.sum() == 0:
                    continue

                # Get gradients for this class
                class_logits = logits[class_mask, class_idx].sum()
                grads = torch.autograd.grad(class_logits, batch_x, retain_graph=True)[0]

                # Accumulate importance scores
                importance_scores[class_idx] += torch.abs(grads[class_mask]).sum(dim=0).cpu().numpy()
                class_counts[class_idx] += class_mask.sum().item()

        # Normalize by class counts
        for class_idx in range(n_classes):
            if class_counts[class_idx] > 0:
                importance_scores[class_idx] /= class_counts[class_idx]

        # Extract top-k markers per class
        markers_per_class = {}
        for class_idx in range(n_classes):
            top_indices = np.argsort(importance_scores[class_idx])[-top_k:][::-1]
            markers_per_class[class_idx] = [gene_names[idx] for idx in top_indices]

        return markers_per_class

    def calculate_consistency(self,
                              markers_1: Dict[int, List[str]],
                              markers_2: Dict[int, List[str]]) -> float:
        """Calculate Jaccard similarity between marker sets"""

        similarities = []

        for class_idx in markers_1.keys():
            if class_idx in markers_2:
                set_1 = set(markers_1[class_idx])
                set_2 = set(markers_2[class_idx])

                if len(set_1) == 0 and len(set_2) == 0:
                    similarity = 1.0
                elif len(set_1) == 0 or len(set_2) == 0:
                    similarity = 0.0
                else:
                    intersection = len(set_1 & set_2)
                    union = len(set_1 | set_2)
                    similarity = intersection / union if union > 0 else 0.0

                similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def analyze_model_consistency(self,
                                  model_class,
                                  datasets: List[str],
                                  model_params: Dict,
                                  n_epochs: int = 50,
                                  top_k: int = 50) -> Tuple[float, Dict]:
        """Analyze cross-dataset consistency for a model"""

        # Train model on each dataset and extract markers
        all_markers = {}

        for dataset in datasets:
            print(f"\nProcessing {dataset}...")

            # Load data
            data = DataProcessor.load_and_preprocess(dataset)
            train_loader, test_loader = DataProcessor.create_dataloaders(data)

            # Initialize model
            if model_class == BioHAN:
                model = model_class(
                    gene_list=data['gene_names'],
                    n_classes=data['n_classes'],
                    **model_params
                ).to(self.device)
            else:
                model = model_class(
                    input_dim=data['n_genes'],
                    n_classes=data['n_classes'],
                    **model_params
                ).to(self.device)

            # Train model
            from core.training import train_model
            train_model(model, train_loader, test_loader, n_epochs=n_epochs, device=self.device)

            # Extract markers
            markers = self.extract_marker_genes(
                model, test_loader, data['gene_names'], data['n_classes'], top_k
            )
            all_markers[dataset] = markers

        # Calculate pairwise consistency
        consistencies = []
        consistency_matrix = {}

        for ds1, ds2 in combinations(datasets, 2):
            consistency = self.calculate_consistency(all_markers[ds1], all_markers[ds2])
            consistencies.append(consistency)
            consistency_matrix[f"{ds1}_vs_{ds2}"] = consistency

        avg_consistency = np.mean(consistencies) if consistencies else 0.0

        return avg_consistency, consistency_matrix

    def compare_methods(self,
                        datasets: List[str],
                        methods: Dict[str, Tuple],
                        n_epochs: int = 50,
                        top_k: int = 50,
                        save_path: Optional[str] = None) -> Dict:
        """Compare cross-dataset consistency across different methods"""

        results = {}

        for method_name, (model_class, params) in methods.items():
            print(f"\n{'=' * 60}")
            print(f"Analyzing {method_name}")
            print('=' * 60)

            avg_consistency, consistency_matrix = self.analyze_model_consistency(
                model_class, datasets, params, n_epochs, top_k
            )

            results[method_name] = {
                'average_consistency': float(avg_consistency),
                'consistency_matrix': consistency_matrix
            }

            print(f"\n{method_name} - Average consistency: {avg_consistency:.3f}")

        # Save results
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)

        return results


def run_consistency_analysis():
    """Run cross-dataset consistency analysis"""

    analyzer = CrossDatasetConsistency()

    # Define datasets to compare
    datasets = [
        'pbmc_multimodal_nygc',
        'lung_atlas_core',
        'kidney_atlas'
    ]

    # Define methods to compare
    methods = {
        'BioHAN': (BioHAN, {'hidden_dim': 256, 'n_modules': 100}),
        'SimpleNN': (SimpleNN, {'hidden_dim': 256}),
        'VAE': (VAEBaseline, {'hidden_dim': 256, 'latent_dim': 64})
    }

    # Run analysis
    results = analyzer.compare_methods(
        datasets=datasets,
        methods=methods,
        n_epochs=30,  # Reduced for faster analysis
        top_k=50,
        save_path='cross_dataset_consistency_results.json'
    )

    # Print summary
    print("\n" + "=" * 60)
    print("CROSS-DATASET CONSISTENCY SUMMARY")
    print("=" * 60)

    for method, result in results.items():
        print(f"{method}: {result['average_consistency']:.3f}")

    return results


if __name__ == "__main__":
    run_consistency_analysis()