"""
Unit tests for BioHAN
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.models import BioHAN
from core.data_processing import DataProcessor, SingleCellDataset
from core.biological_knowledge import BiologicalKnowledgeBase
from baselines.baseline_models import SimpleNN, VAEBaseline


class TestBioHAN:
    """Test BioHAN model"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        n_cells = 100
        n_genes = 500
        n_classes = 5

        X = np.random.randn(n_cells, n_genes).astype(np.float32)
        y = np.random.randint(0, n_classes, n_cells)
        gene_names = [f"Gene_{i}" for i in range(n_genes)]

        return {
            'X': X,
            'y': y,
            'gene_names': gene_names,
            'n_classes': n_classes
        }

    def test_model_initialization(self, sample_data):
        """Test model initialization"""
        model = BioHAN(
            gene_list=sample_data['gene_names'],
            hidden_dim=128,
            n_classes=sample_data['n_classes'],
            n_modules=10,
            n_layers=2
        )

        assert model.n_genes == len(sample_data['gene_names'])
        assert model.n_classes == sample_data['n_classes']

    def test_forward_pass(self, sample_data):
        """Test forward pass"""
        model = BioHAN(
            gene_list=sample_data['gene_names'],
            hidden_dim=128,
            n_classes=sample_data['n_classes']
        )

        x = torch.FloatTensor(sample_data['X'][:10])  # Small batch
        outputs = model(x)

        assert 'logits' in outputs
        assert outputs['logits'].shape == (10, sample_data['n_classes'])
        assert 'marker_scores' in outputs
        assert 'features' in outputs

    def test_interpretations(self, sample_data):
        """Test interpretation extraction"""
        model = BioHAN(
            gene_list=sample_data['gene_names'],
            hidden_dim=128,
            n_classes=sample_data['n_classes']
        )

        x = torch.FloatTensor(sample_data['X'][:10])
        interpretations = model.get_interpretations(x)

        assert 'top_markers' in interpretations
        assert 'predictions' in interpretations
        assert len(interpretations['predictions']) == 10


class TestDataProcessing:
    """Test data processing functions"""

    def test_gene_symbol_extraction(self):
        """Test gene symbol extraction"""
        # Test different formats
        assert DataProcessor.extract_gene_symbol("CD3D_ENSG00000167286") == "CD3D"
        assert DataProcessor.extract_gene_symbol("ENSG00000167286") == "ENSG00000167286"
        assert DataProcessor.extract_gene_symbol("CD3D") == "CD3D"

    def test_dataset_creation(self):
        """Test dataset creation"""
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 5, 100)

        dataset = SingleCellDataset(X, y)
        assert len(dataset) == 100

        x_sample, y_sample = dataset[0]
        assert x_sample.shape == (50,)
        assert isinstance(y_sample, torch.LongTensor)


class TestBiologicalKnowledge:
    """Test biological knowledge base"""

    def test_knowledge_base_initialization(self):
        """Test knowledge base initialization"""
        kb = BiologicalKnowledgeBase()

        assert len(kb.pathways) > 0
        assert len(kb.celltype_markers) > 0
        assert 'T_cell_activation' in kb.pathways
        assert 'T_cells' in kb.celltype_markers

    def test_interaction_matrix(self):
        """Test interaction matrix generation"""
        kb = BiologicalKnowledgeBase()
        gene_list = ['CD3D', 'CD3E', 'CD4', 'CD8A', 'RANDOM1', 'RANDOM2']

        matrix = kb.get_interaction_matrix(gene_list)

        assert matrix.shape == (6, 6)
        assert np.all(matrix >= 0) and np.all(matrix <= 1)
        # CD3D and CD3E should have high interaction
        assert matrix[0, 1] > 0.5


class TestBaselineModels:
    """Test baseline models"""

    def test_simple_nn(self):
        """Test SimpleNN"""
        model = SimpleNN(input_dim=100, hidden_dim=64, n_classes=5)
        x = torch.randn(10, 100)

        output = model(x)
        assert output.shape == (10, 5)

        output_dict = model(x, return_features=True)
        assert 'logits' in output_dict
        assert 'features' in output_dict

    def test_vae_baseline(self):
        """Test VAE baseline"""
        model = VAEBaseline(input_dim=100, hidden_dim=64, latent_dim=32, n_classes=5)
        x = torch.randn(10, 100)

        outputs = model(x)
        assert 'recon_x' in outputs
        assert 'mu' in outputs
        assert 'logvar' in outputs
        assert 'logits' in outputs

        # Test loss computation
        y = torch.randint(0, 5, (10,))
        loss = model.loss(outputs, x, y)
        assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])