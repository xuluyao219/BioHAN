# BioHAN

**Biologically-Informed Hierarchical Attention Networks for Single-Cell RNA-seq Analysis**

BioHAN achieves **94.5% cross-dataset marker gene consistency** — a 26% improvement over standard neural networks — while maintaining competitive classification accuracy (92.7%).

## Key Features

- Discovers marker genes that generalize across different studies
- Integrates biological knowledge (protein interactions, pathways)
- Interpretable results with biologically meaningful markers
- Efficient: 7-10× fewer parameters than transformer models
- Compatible with standard single-cell analysis pipelines

## Installation

```bash
# Clone repository
git clone https://github.com/xuluyao219/BioHAN.git
cd BioHAN

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Download Data

All datasets are from [CZ CELLxGENE](https://cellxgene.cziscience.com/). Run the following to see download instructions:

```bash
python scripts/download_data.py
```

### 2. Run Experiments

```python
# Quick test
python main.py --mode quick --datasets pbmc_multimodal_nygc --epochs 10

# Standard experiments
python main.py --mode standard

# Cross-dataset consistency analysis
python main.py --mode consistency
```

### 3. Basic Usage

```python
from core.models import BioHAN
from core.data_processing import DataProcessor

# Load data
data = DataProcessor.load_and_preprocess('pbmc_multimodal_nygc')
train_loader, test_loader = DataProcessor.create_dataloaders(data)

# Initialize model
model = BioHAN(
    gene_list=data['gene_names'],
    n_classes=data['n_classes']
)

# Train
from core.training import train_model
history = train_model(model, train_loader, test_loader, n_epochs=50)

# Get marker genes
interpretations = model.get_interpretations(test_data)
print(interpretations['top_markers'])
```

## Datasets

| Dataset | Cells | Cell Types | CELLxGENE Collection |
|---------|-------|------------|----------------------|
| PBMC Multimodal | 161,764 | 30 | nygc multimodal pbmc |
| Lung Atlas Core | ~580K | 58 | An integrated cell atlas of the human lung in health and disease (core) |
| Breast Cancer Atlas | 100,064 | 11 | A single-cell and spatially-resolved atlas of human breast cancers |
| Kidney Atlas | ~300K | 28 | Single-cell RNA-seq of the Adult Human Kidney (Version 1.5) |
| Pancreas Islet | ~220K | 14 | Integrated transcriptomes of the 14 pancreatic islet cell types |
| COVID COMBAT | ~650K | 29 | COMBAT project: single cell gene expression data |
| Endothelial | 73,195 | 36 | Tabula Sapiens - Endothelium |

## Performance

### Cross-Dataset Consistency (Main Result)

| Method | Accuracy | Consistency |
|--------|----------|-------------|
| **BioHAN** | 92.7% | **94.5%** |
| SimpleNN | 93.1% | 75.0% |
| VAE | 91.0% | 78.0% |
| scVI | 91.6% | 82.3% |

### Comparison with State-of-the-Art

| Method | PBMC Accuracy | Parameters | Consistency |
|--------|---------------|------------|-------------|
| BioHAN | 92.6% | 12.3M | ✓ (94.5%) |
| scBERT | 92.0% | 86.7M | Not reported |
| scGPT | 91.8% | 124.4M | Not reported |
| Geneformer | 91.2% | 95.2M | Not reported |

## Model Architecture

BioHAN combines five key components:

1. **Gene Expression Encoder**: Initial feature extraction
2. **Gene Regulatory Modules**: Learns functional gene groups
3. **Biologically-Constrained Attention**: Uses protein interaction networks
4. **Hierarchical Cell States**: Models cell type relationships
5. **Temperature-Scaled Markers**: Sharp selection of marker genes

## Citation

If you use BioHAN, please cite:

```bibtex
@article{biohan2025,
  title={BioHAN: Biologically-Informed Hierarchical Attention Networks for Cross-Dataset Reproducible Cell Type Discovery},
  author={Xulu Yao and others},
  journal={bioRxiv},
  year={2025}
}
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- scanpy 1.9+
- numpy, pandas, scikit-learn
- 16GB RAM (32GB recommended)
- GPU optional but recommended

## License

MIT License - see [LICENSE](LICENSE) file.

## Contact

- GitHub Issues: [https://github.com/xuluyao219/BioHAN/issues](https://github.com/xuluyao219/BioHAN/issues)
- Email: xulu.yao@ieee.org
