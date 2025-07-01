
"""
Download instructions for BioHAN datasets from CZ CELLxGENE
"""

import os
import argparse
from pathlib import Path

# Dataset information
DATASETS = {
    'pbmc_multimodal_nygc': {
        'collection': 'nygc multimodal pbmc',
        'cells': 161764,
        'size_mb': 800
    },
    'lung_atlas_core': {
        'collection': 'An integrated cell atlas of the human lung in health and disease (core)',
        'cells': 584944,
        'size_mb': 2500
    },
    'breast_cancer_atlas': {
        'collection': 'A single-cell and spatially-resolved atlas of human breast cancers',
        'cells': 100064,
        'size_mb': 600
    },
    'kidney_atlas': {
        'collection': 'Single-cell RNA-seq of the Adult Human Kidney (Version 1.5)',
        'cells': 300856,
        'size_mb': 1800
    },
    'pancreas_islet': {
        'collection': 'Integrated transcriptomes of the 14 pancreatic islet cell types',
        'cells': 220621,
        'size_mb': 1200
    },
    'covid_combat': {
        'collection': 'COMBAT project: single cell gene expression data from COVID-19, sepsis and flu patient PBMCs',
        'cells': 650487,
        'size_mb': 3000
    },
    'endothelial_multitissue': {
        'collection': 'Tabula Sapiens - Endothelium',
        'cells': 73195,
        'size_mb': 400
    }
}


def show_instructions():
    """Show download instructions"""
    print("\nBioHAN Dataset Download Instructions")
    print("=" * 60)
    print("\nHow to download:")
    print("1. Visit https://cellxgene.cziscience.com/")
    print("2. Search for the collection name (see below)")
    print("3. Download in .h5ad format")
    print("4. Save to real_datasets/ folder with exact filename")
    print("\nDatasets needed:")
    print("-" * 60)

    for name, info in DATASETS.items():
        print(f"\nDataset: {name}.h5ad")
        print(f"Collection: {info['collection']}")
        print(f"Cells: {info['cells']:,}")
        print(f"Size: ~{info['size_mb']} MB")


def check_datasets(data_dir='real_datasets'):
    """Check which datasets exist"""
    print(f"\nChecking datasets in {data_dir}/")
    print("=" * 60)

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Directory {data_dir} does not exist!")
        return

    found = 0
    missing = 0

    for name in DATASETS:
        filename = f"{name}.h5ad"
        filepath = data_path / filename

        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ {filename:<35} ({size_mb:.1f} MB)")
            found += 1
        else:
            print(f"✗ {filename:<35} (MISSING)")
            missing += 1

    print("-" * 60)
    print(f"Found: {found}/{len(DATASETS)} datasets")

    if missing > 0:
        print("\nRun 'python download_data.py --instructions' to see how to download")


def main():
    parser = argparse.ArgumentParser(description='BioHAN dataset download helper')
    parser.add_argument('--instructions', action='store_true',
                        help='Show download instructions')
    parser.add_argument('--check', action='store_true',
                        help='Check which datasets are downloaded')
    parser.add_argument('--data-dir', default='real_datasets',
                        help='Dataset directory (default: real_datasets)')

    args = parser.parse_args()

    if args.instructions:
        show_instructions()
    elif args.check:
        check_datasets(args.data_dir)
    else:
        # Default: show both
        show_instructions()
        print("\n" + "=" * 60)
        check_datasets(args.data_dir)


if __name__ == '__main__':
    main()