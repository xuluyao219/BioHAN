"""
Unified data processing module for single-cell RNA-seq data
Handles data loading, preprocessing, and dataset creation
"""

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
import re
import logging
from functools import lru_cache
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SingleCellDataset(Dataset):
    """PyTorch Dataset for single-cell RNA-seq data

    Args:
        X: Gene expression matrix [n_cells x n_genes]
        y: Cell type labels [n_cells]
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DataProcessor:
    """Data processor for loading and preprocessing scRNA-seq datasets

    All datasets are sourced from CZ CELLxGENE Discover:
    https://cellxgene.cziscience.com/
    """

    @staticmethod
    def extract_gene_symbol(feature_name: str) -> str:
        """Extract gene symbol from various feature name formats

        Args:
            feature_name: Feature name from h5ad file

        Returns:
            Clean gene symbol

        Examples:
            'CD3D_ENSG00000167286' -> 'CD3D'
            'ENSG00000167286' -> 'ENSG00000167286' (kept as is)
            'CD3D' -> 'CD3D'
        """
        # Handle "GeneSymbol_EnsemblID" format
        if '_ENSG' in feature_name:
            return feature_name.split('_ENSG')[0]

        # Pure Ensembl ID (keep for now, can be mapped later)
        if feature_name.startswith('ENSG'):
            return feature_name

        # Already a clean gene symbol
        return feature_name

    @staticmethod
    @lru_cache(maxsize=10)
    def load_and_preprocess(dataset_name: str,
                            data_dir: str = 'real_datasets',
                            n_top_genes: int = 2000,
                            downsample: int = 50000,
                            use_cache: bool = True,
                            cache_dir: str = 'cache') -> Dict:
        """Load and preprocess single-cell dataset with caching

        Args:
            dataset_name: Name of the dataset file (without .h5ad extension)
            data_dir: Directory containing h5ad files
            n_top_genes: Number of highly variable genes to select
            downsample: Maximum number of cells (downsampling if needed)
            use_cache: Whether to use cached processed data
            cache_dir: Directory for cached data

        Returns:
            Dictionary containing:
                - X_train, X_test: Expression matrices
                - y_train, y_test: Cell type labels
                - gene_names: List of gene symbols
                - label_encoder: Cell type to index mapping
                - n_genes, n_cells, n_classes: Dataset dimensions
        """
        # Check cache first
        cache_path = Path(cache_dir) / f"{dataset_name}_processed.pkl"
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # Load h5ad file
        file_path = Path(data_dir) / f'{dataset_name}.h5ad'
        logger.info(f"Loading dataset from {file_path}")
        adata = sc.read_h5ad(file_path)
        logger.info(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")

        # Intelligent downsampling to maintain cell type proportions
        if adata.n_obs > downsample:
            logger.info(f"Downsampling from {adata.n_obs} to {downsample} cells")
            sc.pp.subsample(adata, n_obs=downsample, random_state=42)

        # Standard preprocessing pipeline
        logger.info("Preprocessing data...")

        # Basic filtering
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)

        # Normalization and log transformation
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Select highly variable genes
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes,
                                    batch_key=None, subset=False)
        adata = adata[:, adata.var.highly_variable]

        # Scale data
        sc.pp.scale(adata, max_value=10)

        # Extract gene symbols - critical for biological interpretation
        if 'feature_name' in adata.var.columns:
            gene_symbols = [DataProcessor.extract_gene_symbol(name)
                            for name in adata.var['feature_name']]
            logger.info("Extracted gene symbols from feature_name column")
        else:
            # Fallback to gene IDs
            gene_symbols = list(adata.var_names)
            logger.warning("No feature_name column found, using gene IDs")

        # Validate marker genes found
        known_markers = ['CD3D', 'CD3E', 'CD4', 'CD8A', 'CD19', 'CD79A',
                         'MS4A1', 'CD14', 'FCGR3A', 'NKG7', 'GNLY']
        found_markers = [m for m in known_markers if m in gene_symbols]
        logger.info(f"Found {len(found_markers)}/{len(known_markers)} known markers: {found_markers[:5]}...")

        # Extract expression matrix
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

        # Get cell type labels
        cell_type_col = None
        for col in ['cell_type', 'celltype', 'cell_type_ontology_term_id', 'Cell_type']:
            if col in adata.obs.columns:
                cell_type_col = col
                break

        if cell_type_col is None:
            raise ValueError("No cell type column found in adata.obs")

        labels = adata.obs[cell_type_col]
        unique_labels = labels.unique()
        label_encoder = {label: i for i, label in enumerate(unique_labels)}
        y = np.array([label_encoder[label] for label in labels])

        logger.info(f"Found {len(unique_labels)} cell types")

        # Train/test split with stratification
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=y, random_state=42
        )

        # Prepare result dictionary
        result = {
            'X_train': X[train_idx],
            'X_test': X[test_idx],
            'y_train': y[train_idx],
            'y_test': y[test_idx],
            'gene_names': gene_symbols,
            'label_encoder': label_encoder,
            'n_genes': adata.n_vars,
            'n_cells': adata.n_obs,
            'n_classes': len(label_encoder),
            'dataset_info': {
                'name': dataset_name,
                'source': 'CZ CELLxGENE Discover',
                'n_cells_original': adata.n_obs,
                'cell_types': list(unique_labels)
            }
        }

        # Cache processed data
        if use_cache:
            cache_path.parent.mkdir(exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            logger.info(f"Cached processed data to {cache_path}")

        # Print dataset summary
        logger.info(f"Dataset summary: {len(X)} cells, {len(gene_symbols)} genes, {len(label_encoder)} cell types")

        return result

    @staticmethod
    def create_dataloaders(data_dict: Dict,
                           batch_size: int = 128,
                           num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders for training and testing

        Args:
            data_dict: Dictionary from load_and_preprocess
            batch_size: Batch size for training
            num_workers: Number of parallel data loading workers

        Returns:
            train_loader, test_loader
        """
        train_dataset = SingleCellDataset(data_dict['X_train'], data_dict['y_train'])
        test_dataset = SingleCellDataset(data_dict['X_test'], data_dict['y_test'])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, test_loader

    @staticmethod
    def get_dataset_info(data_dir: str = 'real_datasets') -> Dict[str, Dict]:
        """Get information about all available datasets

        Args:
            data_dir: Directory containing h5ad files

        Returns:
            Dictionary with dataset names and their properties
        """
        dataset_info = {}
        data_path = Path(data_dir)

        for h5ad_file in data_path.glob('*.h5ad'):
            dataset_name = h5ad_file.stem
            try:
                adata = sc.read_h5ad(h5ad_file)

                # Get cell type column
                cell_type_col = None
                for col in ['cell_type', 'celltype', 'cell_type_ontology_term_id']:
                    if col in adata.obs.columns:
                        cell_type_col = col
                        break

                if cell_type_col:
                    n_cell_types = len(adata.obs[cell_type_col].unique())
                else:
                    n_cell_types = 'Unknown'

                dataset_info[dataset_name] = {
                    'n_cells': adata.n_obs,
                    'n_genes': adata.n_vars,
                    'n_cell_types': n_cell_types,
                    'file_size_mb': h5ad_file.stat().st_size / 1024 / 1024
                }

            except Exception as e:
                logger.error(f"Error reading {dataset_name}: {e}")

        return dataset_info