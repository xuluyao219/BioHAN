"""
Unified experiment runner with UTF-8 encoding support
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime

from core.models import BioHAN
from core.data_processing import DataProcessor
from core.training import train_model
from core.evaluation import evaluate_model
from baselines.baseline_models import VAEBaseline, SimpleNN, get_sklearn_baselines


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


class ExperimentRunner:
    """Experiment runner with proper encoding support"""

    def __init__(self, output_dir: str = 'results'):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(output_dir) / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run_all_experiments(self,
                            datasets: List[str],
                            methods: List[str] = ['biohan', 'vae', 'simple_nn'],
                            n_epochs: int = 50) -> Dict:
        """Run all experiments"""
        all_results = {}

        for dataset in datasets:
            print(f"\n{'=' * 60}\nDataset: {dataset}\n{'=' * 60}")

            try:
                # Load data
                data = DataProcessor.load_and_preprocess(dataset)
                train_loader, test_loader = DataProcessor.create_dataloaders(data)
            except Exception as e:
                print(f"Error loading dataset {dataset}: {e}")
                continue

            dataset_results = {}

            # Run each method
            for method in methods:
                print(f"\nRunning {method}...")

                try:
                    if method == 'biohan':
                        results = self.run_biohan(data, train_loader, test_loader, n_epochs)
                    elif method == 'vae':
                        results = self.run_vae(data, train_loader, test_loader, n_epochs)
                    elif method == 'simple_nn':
                        results = self.run_simple_nn(data, train_loader, test_loader, n_epochs)
                    elif method in ['random_forest', 'svm']:
                        results = self.run_sklearn(data, method)
                    else:
                        continue

                    dataset_results[method] = results
                    print(f"{method} - ARI: {results['metrics']['ari']:.3f}, "
                          f"Accuracy: {results['metrics']['accuracy']:.3f}")
                except Exception as e:
                    print(f"Error running {method}: {e}")
                    continue

            all_results[dataset] = dataset_results

            # Save intermediate results
            self._save_results(dataset, dataset_results)

        # Generate report
        self._generate_report(all_results)

        return all_results

    def run_biohan(self, data: Dict, train_loader, test_loader, n_epochs: int) -> Dict:
        """Run BioHAN model"""
        model = BioHAN(
            gene_list=data['gene_names'],
            hidden_dim=512,
            n_classes=data['n_classes'],
            n_modules=150,
            n_layers=4,
            dropout=0.3
        ).to(self.device)

        # Train
        history = train_model(
            model, train_loader, test_loader,
            n_epochs=n_epochs,
            learning_rate=5e-4,
            device=self.device
        )

        # Evaluate
        metrics = evaluate_model(model, test_loader, self.device)

        # Get interpretations
        model.eval()
        sample_x, _ = next(iter(test_loader))
        interpretations = model.get_interpretations(sample_x.to(self.device))

        return {
            'metrics': metrics,
            'history': history,
            'interpretations': interpretations
        }

    def run_vae(self, data: Dict, train_loader, test_loader, n_epochs: int) -> Dict:
        """Run VAE baseline"""
        model = VAEBaseline(
            input_dim=data['n_genes'],
            n_classes=data['n_classes']
        ).to(self.device)

        # Custom training loop for VAE
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(n_epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = model.loss(outputs, batch_x, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        metrics = evaluate_model(model, test_loader, self.device,
                                 feature_key='z', logit_key='logits')

        return {'metrics': metrics}

    def run_simple_nn(self, data: Dict, train_loader, test_loader, n_epochs: int) -> Dict:
        """Run simple neural network"""
        model = SimpleNN(
            input_dim=data['n_genes'],
            n_classes=data['n_classes']
        ).to(self.device)

        # Train
        train_model(model, train_loader, test_loader, n_epochs=n_epochs, device=self.device)

        # Evaluate
        metrics = evaluate_model(model, test_loader, self.device)

        return {'metrics': metrics}

    def run_sklearn(self, data: Dict, method: str) -> Dict:
        """Run sklearn baseline"""
        models = get_sklearn_baselines(data['n_classes'])
        model = models[method]

        # Train
        model.fit(data['X_train'], data['y_train'])

        # Predict
        y_pred = model.predict(data['X_test'])

        # Calculate metrics
        from sklearn.metrics import adjusted_rand_score, accuracy_score
        metrics = {
            'ari': adjusted_rand_score(data['y_test'], y_pred),
            'accuracy': accuracy_score(data['y_test'], y_pred)
        }

        return {'metrics': metrics}

    def _save_results(self, dataset: str, results: Dict):
        """Save results with UTF-8 encoding"""
        save_path = self.output_dir / f'{dataset}_results.json'

        # Prepare serializable results
        serializable = {}
        for method, result in results.items():
            serializable[method] = {
                'metrics': self._convert_metrics(result['metrics'])
            }
            if 'interpretations' in result:
                # Save top 10 marker genes
                top_markers = {}
                for cls, markers in result['interpretations'].get('top_markers', {}).items():
                    top_markers[cls] = [(gene, float(score)) for gene, score in markers[:10]]
                serializable[method]['top_markers'] = top_markers

        # Save with UTF-8 encoding
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, cls=NumpyEncoder)

    def _convert_metrics(self, metrics: Dict) -> Dict:
        """Convert metrics to serializable format"""
        converted = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating, np.float32, np.float64)):
                converted[key] = float(value)
            elif isinstance(value, (int, float, str)):
                converted[key] = value
            elif isinstance(value, np.ndarray):
                converted[key] = value.tolist()
            else:
                # Skip non-serializable values
                continue
        return converted

    def _generate_report(self, results: Dict):
        """Generate report with UTF-8 encoding"""
        report_path = self.output_dir / 'report.md'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Experiment Report\n\n")
            f.write(f"Generated: {self.timestamp}\n\n")

            # Performance table
            f.write("## Performance Summary\n\n")
            f.write("| Dataset | BioHAN | VAE | Simple NN | Random Forest | SVM |\n")
            f.write("|---------|--------|-----|-----------|---------------|-----|\n")

            for dataset, dataset_results in results.items():
                row = f"| {dataset} "
                for method in ['biohan', 'vae', 'simple_nn', 'random_forest', 'svm']:
                    if method in dataset_results:
                        ari = dataset_results[method]['metrics']['ari']
                        row += f"| {ari:.3f} "
                    else:
                        row += "| - "
                f.write(row + "|\n")

        print(f"\nReport saved to: {report_path}")